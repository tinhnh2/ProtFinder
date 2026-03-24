#!/usr/bin/env python3
"""
Generate simulated MSAs using IQ-TREE's Ali-Sim with parameters sampled from empirical distributions.

This script generates simulated MSAs by:
1. Loading inverse CDF distributions from .npz files
2. Generating random tree topologies
3. Sampling branch lengths and model parameters from empirical distributions
4. Running IQ-TREE Ali-Sim to generate MSAs

Usage:
    # Generate training/validation set
    python data_preparation/simulation.py \
        --iqtree_path /path/to/iqtree3.exe \
        --param_dir fitted_empirical_dist \
        --trees_dir ./simulated_trees \
        --output_dir ./simulated_alignments \
        --data_type train_val \
        --num_iterations 200
    
    # Generate test set
    python data_preparation/simulation.py \
        --iqtree_path /path/to/iqtree3.exe \
        --param_dir fitted_empirical_dist \
        --trees_dir ./simulated_trees \
        --output_dir ./simulated_alignments \
        --data_type test \
        --num_iterations 50
"""

import argparse
import subprocess
from pathlib import Path
from ete3 import Tree
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preparation.empirical_dist import EmpiricalInverseCDF


MODEL_MAP = {'LG': 0, 'WAG': 1, 'JTT': 2, 'Q.plant': 3, 'Q.bird': 4, 'Q.mammal': 5, 'Q.pfam': 6}


def preload_inverse_cdfs(param_dir):
    """
    Preload all inverse CDF distributions from a directory.
    
    Args:
        param_dir: Directory containing .npz files
    
    Returns:
        Dictionary mapping parameter names to EmpiricalInverseCDF instances
    """
    param_dir = Path(param_dir)
    # Mapping of parameter names to EmpiricalInverseCDF instances
    cdf_dict = {}
    
    for file in sorted(param_dir.glob('*.npz')):
        key = file.stem
        cdf_dict[key] = EmpiricalInverseCDF.load(str(file))
    
    return cdf_dict


def sample_from_inverse(cdf_dict, key, lbound, ubound, r):
    """
    Sample a value from inverse CDF distribution within bounds.
    
    Args:
        cdf_dict: Dictionary of EmpiricalInverseCDF instances
        key: Parameter name
        lbound: Lower bound for sampling
        ubound: Upper bound for sampling
        r: Number of decimal places to round
    
    Returns:
        Sampled value
    """
    
    dist = cdf_dict[key]
    
    # Sample from the inverse CDF
    sample = round(float(dist.rvs(1)[0]), r)
    
    # Ensure bounds
    while (sample <= lbound or sample >= ubound):
        sample = round(float(dist.rvs(1)[0]), r)

    return sample


def generate_alignment(
    model_name: str,
    F: bool,
    I: bool,
    G4: bool,
    cdf_dict: dict,
    n_taxa: int,
    n_sites: int,
    iteration: int,
    seed: int,
    iqtree_path: str,
    trees_dir: Path,
    output_dir: Path
):
    """
    Generate a single simulated alignment.
    
    Args:
        model_name: Base substitution model name
        F: Whether to use +F (empirical frequencies)
        I: Whether to use +I (invariant sites)
        G4: Whether to use +G4 (Gamma distribution)
        cdf_dict: Dictionary of inverse CDF functions
        n_taxa: Number of taxa
        n_sites: Sequence length
        iteration: Iteration number
        seed: Random seed
        iqtree_path: Path to IQ-TREE executable (iqtree3.exe)
        trees_dir: Directory to save tree files
        output_dir: Directory to save alignment files
    """
    trees_dir = Path(trees_dir)
    output_dir = Path(output_dir)
    trees_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate random tree topology
    tree_prefix = trees_dir / "gen_tree"
    tree_cmd = [
        iqtree_path,
        "--alisim", "gen_tree",
        "-t", f"RANDOM{{yh/{n_taxa}}}",
        "-m", "LG",
        "--prefix", str(tree_prefix),
        "-seed", str(seed),
        "-redo"
    ]
    
    try:
        subprocess.run(
            tree_cmd,
            cwd=str(trees_dir),
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error generating tree: {e}")
        print(f"Command: {' '.join(tree_cmd)}")
        raise
    
    tree_file = tree_prefix.with_suffix('.treefile')
    if not tree_file.exists():
        raise FileNotFoundError(f"Tree file not created: {tree_file}")
    
    # Step 2: Modify branch lengths    
    tree = Tree(open(tree_file).read())
    for node in tree.traverse():
        if not node.is_root():
            key = 'external' if node.is_leaf() else 'internal'
            node.dist = sample_from_inverse(cdf_dict, key, 0, 1, 10)
    
    # Step 3: Assemble model string
    file_name = f"{MODEL_MAP[model_name]}({n_taxa}){model_name}"
    command = f"-m {model_name}"
    
    if F:
        freqs = []
        for aa in 'ARNDCQEGHILKMFPSTWYV':
            key = f'FREQ_{aa}'
            freq = sample_from_inverse(cdf_dict, key, 0, 1, 5)
            freqs.append(str(freq))
        command += '+F{' + '/'.join(freqs) + '}'
        file_name += '+F'
    
    if I:
        key = 'I'
        invariant = sample_from_inverse(cdf_dict, key, 0, 0.9, 5)
        command += f'+I{{{invariant}}}'
        file_name += f'+I{{{invariant}}}'
    
    if G4:
        key = 'G4'
        alpha = sample_from_inverse(cdf_dict, key, 0.001, 10, 5)
        command += f'+G4{{{alpha}}}'
        file_name += f'+G4{{{alpha}}}'
    
    file_name += f'[{n_sites}]_{iteration}'
    
    # Step 4: Save modified tree
    tree_out = trees_dir / f"{file_name}.treefile"
    with open(tree_out, 'w') as f:
        f.write(tree.write(format=1) + '\n')
    
    # Step 5: Run IQ-TREE simulation
    # Use absolute path for tree file
    tree_out_abs = tree_out.resolve()
    full_command = [
        iqtree_path,
        "--alisim", file_name,
        "-t", str(tree_out_abs),
    ] + command.split() + [
        "--length", str(n_sites),
        "-seed", str(seed)
    ]
    
    try:
        result = subprocess.run(
            full_command,
            cwd=str(output_dir),
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running IQ-TREE simulation: {e}")
        print(f"Command: {' '.join(full_command)}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        raise


def generate_all(
    base_model_list: list,
    cdf_dict: dict,
    num_iterations: int,
    iqtree_path: str,
    trees_dir: Path,
    output_dir: Path
):
    """
    Generate all combinations of simulated alignments.
    
    Args:
        base_model_list: List of base substitution models
        cdf_dict: Dictionary of inverse CDF functions
        num_iterations: Number of iterations
        iqtree_path: Path to IQ-TREE executable (iqtree3.exe)
        trees_dir: Directory to save tree files
        output_dir: Directory to save alignment files
    """
    combo_id = 0
    total_combinations = (
        len(base_model_list) * 
        4 *  # n_sites
        6 *  # n_taxa
        2 *  # F
        2 *  # I
        2 *  # G4
        num_iterations
    )
    
    print(f"Generating {total_combinations} alignments")
    
    for i in range(num_iterations):
        for base_model in base_model_list:
            for n_sites in [100, 500, 1000, 3000]:
                for n_taxa in [8, 16, 32, 64, 128, 256]:
                    for F in [True, False]:
                        for I in [True, False]:
                            for G4 in [True, False]:
                                seed = 1000000 + i * 2000 + combo_id
                                generate_alignment(
                                    base_model, F, I, G4, cdf_dict, n_taxa, n_sites, i, seed, iqtree_path, trees_dir, output_dir
                                )
                                combo_id += 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulated MSAs using IQ-TREE Ali-Sim"
    )
    parser.add_argument(
        "--iqtree_path",
        type=str,
        required=True,
        help="Path to IQ-TREE executable (iqtree3.exe)"
    )
    parser.add_argument(
        "--param_dir",
        type=str,
        required=True,
        help="Directory containing inverse CDF .npz files"
    )
    parser.add_argument(
        "--trees_dir",
        type=str,
        default="./simulated_trees",
        help="Directory to save tree files (default: ./simulated_trees)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./simulated_alignments",
        help="Directory to save alignment files (default: ./simulated_alignments)"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=200,
        help="Number of iterations (default: 200)"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["train_val", "test"],
        default="train_val",
        help="Data type: 'train_val' for training/validation set, 'test' for test set (default: train_val). Automatically appends suffix to trees_dir and output_dir."
    )
    
    args = parser.parse_args()
    
    iqtree_path = Path(args.iqtree_path)
    
    # Automatically add suffix to paths based on data_type
    trees_dir_base = Path(args.trees_dir)
    output_dir_base = Path(args.output_dir)
    
    suffix = f"_{args.data_type}"
    if not trees_dir_base.name.endswith(suffix):
        trees_dir = trees_dir_base.parent / f"{trees_dir_base.name}{suffix}"
    else:
        trees_dir = trees_dir_base
    
    if not output_dir_base.name.endswith(suffix):
        output_dir = output_dir_base.parent / f"{output_dir_base.name}{suffix}"
    else:
        output_dir = output_dir_base
    
    # Load inverse CDFs
    cdf_dict = preload_inverse_cdfs(args.param_dir)
    
    # Generate alignments
    base_models = list(MODEL_MAP.keys())
    generate_all(base_models, cdf_dict, args.num_iterations, str(iqtree_path), trees_dir, output_dir)


if __name__ == '__main__':
    main()
