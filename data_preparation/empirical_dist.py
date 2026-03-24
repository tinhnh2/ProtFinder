#!/usr/bin/env python3
"""
Fit empirical inverse CDF distributions from real MSA parameter distributions.

This script reads CSV files containing parameters extracted from real MSAs,
fits empirical inverse CDFs using spline interpolation, and saves them as .npz files.

Usage:
    python data_preparation/empirical_dist.py \
        --input_file empirical_parameters/F_parameters.csv \
        --output_dir fitted_empirical_dist
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import interpolate


class EmpiricalInverseCDF:
    """
    Fit and sample from an empirical inverse CDF using spline interpolation.
    
    Uses scipy's spline interpolation to approximate the inverse CDF from empirical data.
    """
    
    def __init__(self, data):
        """
        Args:
            data: Array-like data to fit the inverse CDF
        """
        # Fit spline parameters for the inverse CDF
        data_sorted = np.sort(np.asarray(data))
        y = np.linspace(0, 1, len(data_sorted))
        self._tck = interpolate.splrep(y, data_sorted)

    def rvs(self, size):
        """
        Sample random values from the inverse CDF.
        
        Args:
            size: Number of samples to generate
        
        Returns:
            Array of sampled values
        """
        u = np.random.rand(size)
        return interpolate.splev(u, self._tck)

    def save(self, path):
        """
        Save the spline parameters to a .npz file.
        
        Args:
            path: Path to save the .npz file
        """
        np.savez(path, *self._tck)

    @staticmethod
    def load(path):
        """
        Load spline parameters from a .npz file.
        
        Args:
            path: Path to the .npz file
        
        Returns:
            EmpiricalInverseCDF instance with loaded spline parameters
        """
        data = np.load(path)
        tck = (data['arr_0'], data['arr_1'], data['arr_2'])
        obj = EmpiricalInverseCDF.__new__(EmpiricalInverseCDF)
        obj._tck = tck
        return obj
        


def fit_empirical_inverse_cdf_from_file(file_path, output_dir):
    """
    Fit empirical inverse CDF from a CSV file.
    
    Processes all columns in the CSV file, fitting a distribution for each column.
    
    Args:
        file_path: Path to CSV file
        output_dir: Directory to save .npz files
    
    Returns:
        Dictionary with fitting results
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if file_path.suffix.lower() != '.csv':
        raise ValueError(f"Only CSV files are supported. Got: {file_path.suffix}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading CSV file: {file_path}")
    df = pd.read_csv(file_path)    
    print(f"Found {len(df.columns)} columns to process")
    
    # Process each column
    for column in df.columns:
        column_data = df[column].dropna()
        
        # Fit inverse CDF for this column
        empirical_inverse_cdf = EmpiricalInverseCDF(column_data)
        save_path = output_dir / f"{column}.npz"
        empirical_inverse_cdf.save(save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Fit empirical inverse CDF distributions from CSV parameter files using spline interpolation"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input CSV file with parameters"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fitted_empirical_dist",
        help="Directory to save .npz files (default: fitted_empirical_dist)"
    )
    
    args = parser.parse_args()
    
    fit_empirical_inverse_cdf_from_file(args.input_file, args.output_dir)



if __name__ == '__main__':
    main()
