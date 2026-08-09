# ProtFinder: An efficient machine learning framework for protein model selection on real data

Model selection is a fundamental step in phylogenetic analysis that determines the best-fit model of sequence evolution for a given multiple sequence alignment. Popular model selection methods, such as ModelFinder, rely on statistical information criteria, such as the Bayesian Information Criterion (BIC) or the Akaike Information Criterion (AIC).However, these approaches are computationally expensive and the use of information criteria has been the subject of ongoing discussion. Recently, machine learning has emerged as a promising approach for phylogenetic model selection in both nucleotide and protein sequence analyses. ModelDetector is currently the only machine learning-based method for amino acid substitution model selection. However, because ModelDetector was trained on simulated data, it does not perform well on real datasets. Another limitation is that it does not support different rate heterogeneity across sites (RHAS) models. To overcome these limitations, we introduce ProtFinder, an efficient machine learning framework for protein model selection that predicts amino acid substitution models, RHAS models, and amino acid frequency models. To enable ProtFinder to work with real datasets, we employed a transfer learning strategy consisting of three stages: (1) initial training on large-scale simulated data, (2) joint training on both simulated and real data, and (3) final fine-tuning using real data only. Experimental results show that ProtFinder outperformed ModelDetector in amino acid substitution model selection. ProtFinder achieved comparable accuracy to the maximum likelihood method ModelFinder for substitution model selection on medium and large MSAs. It performs slightly better than ModelFinder in RHAS model selection and substantially outperforms it in amino acid frequency model determination. Notably, ProtFinder is up to 1,400 times faster than ModelFinder in terms of inference time, making it particularly suitable for medium and large datasets.

## Project Overview

The project consists of 7 main steps:

1. **Empirical Distribution Fitting** - Fit inverse CDF distributions from real MSA parameters
2. **Data Simulation** - Generate simulated MSAs using IQ-TREE Ali-Sim
3. **Feature Extraction** - Extract features from simulated MSAs
4. **Feature Packaging** - Package features into HDF5 format
5. **Model Training** - Train and evaluate machine learning models
6. **Fine-Tuning Models** - Fine tune models
7. **Model Testing** - Test the models


## Workflow

### Step 1: Fit Empirical Distributions

**Note**: This step can be skipped because the resulting distributions are already in `fitted_empirical_dist/`.

```bash
# Fit F parameters (amino acid frequencies)
python data_preparation/empirical_dist.py \
    --input_file empirical_parameters/F_parameters.csv \
    --output_dir fitted_empirical_dist

# Fit G4 parameters
python data_preparation/empirical_dist.py \
    --input_file empirical_parameters/G4_parameters.csv \
    --output_dir fitted_empirical_dist

# Fit I parameters
python data_preparation/empirical_dist.py \
    --input_file empirical_parameters/I_parameters.csv \
    --output_dir fitted_empirical_dist

# Fit external branch lengths
python data_preparation/empirical_dist.py \
    --input_file empirical_parameters/external_lengths.csv \
    --output_dir fitted_empirical_dist

# Fit internal branch lengths
python data_preparation/empirical_dist.py \
    --input_file empirical_parameters/internal_lengths.csv \
    --output_dir fitted_empirical_dist
```

### Step 2: Generate Simulated Data

Generate training/validation set and test set separately, because **the test set usually has a different setting**.

```bash
# Generate training/validation set
python data_preparation/simulation.py \
    --iqtree_path iqtree3 \
    --param_dir fitted_empirical_dist \
    --trees_dir ./simulated_trees \
    --output_dir ./simulated_alignments \
    --data_type train_val \
    --num_iterations 200

# Generate test set
python data_preparation/simulation.py \
    --iqtree_path iqtree3 \
    --param_dir fitted_empirical_dist \
    --trees_dir ./simulated_trees \
    --output_dir ./simulated_alignments \
    --data_type test \
    --num_iterations 40
	
# Generate simulated msa set for joint training. Need merge the 15330 real MSAs to this set.
python data_preparation/simulation.py \
    --iqtree_path iqtree3 \
    --param_dir fitted_empirical_dist \
    --trees_dir ./simulated_trees \
    --output_dir ./simulated_alignments_joint \
    --data_type train_val \
    --num_iterations 10
```

**Note**: The `--data_type` parameter automatically appends `_train_val` or `_test` suffix to output directories. For example, `./simulated_alignments` becomes `./simulated_alignments_train_val` or `./simulated_alignments_test`.

### Step 3: Extract Features

Extract features from training/validation and test sets:

```bash
# Extract features from training/validation set
python data_preparation/feature_extraction.py \
    --alignments_dir ./simulated_alignments \
    --output_dir ./extracted_features \
    --data_type train_val \
    --num_workers 7

# Extract features from test set
python data_preparation/feature_extraction.py \
    --alignments_dir ./simulated_alignments \
    --output_dir ./extracted_features \
    --data_type test \
    --num_workers 7

# Extract features from joint_training set
python data_preparation/feature_extraction.py \
    --alignments_dir ./simulated_alignments_joint \
    --output_dir ./extracted_features \
    --data_type train_val \
    --num_workers 7

# Extract features from real set. Unziping the zip hssp1471_best_fit.zip before executing the command.
python data_preparation/feature_extraction.py \
	--alignments_dir ./real_alignments_15330 \
	--output_dir ./extracted_features_tuning \
	--data_type train_val \
	--num_workers 7

# Extract features from real 1471 HSSP test set. Unziping the zip hssp1471_best_fit.zip before executing the command.
python data_preparation/feature_extraction.py \
    --alignments_dir ./hssp1471_best_fit \
    --output_dir ./extracted_features_hssp1471 \
    --data_type test \
    --num_workers 7
```

**Note**: The `--data_type` parameter automatically appends suffix to input and output directories. Make sure to use the same `--data_type` as in Step 2.

### Step 4: Package Features to HDF5

Package features into HDF5 format. For training/validation sets, use `split_mode` to split data into train and val groups. Two splitting methods are provided:

- **Iteration-based splitting** (`split_mode="iteration"`): Splits data based on iteration numbers (e.g., iterations < threshold for training, ≥ threshold for validation). This method ensures perfectly balanced and consistent training/validation sets across all three machine learning models, as each class has the same number of samples per iteration. This was the method used in previous experiments.

- **Random splitting** (`split_mode="random"`): Randomly splits data with a specified ratio. The resulting datasets should be nearly balanced when the dataset is large enough, and will be consistent across all three models if the same random seed and feature filenames are used (which is the default behavior).

```bash
# Split training and validation set and package them
python data_preparation/package_features.py \
    --qfinder_dir ./extracted_features_train_val/QFinder \
    --rhasfinder_dir ./extracted_features_train_val/RHASFinder \
    --ffinder_dir ./extracted_features_train_val/FFinder \
    --output_dir ./hdf5_features \
    --split_mode random \
    --train_ratio 0.8

# Package test set (no splitting)
python data_preparation/package_features.py \
    --qfinder_dir ./extracted_features_test/QFinder \
    --rhasfinder_dir ./extracted_features_test/RHASFinder \
    --ffinder_dir ./extracted_features_test/FFinder \
    --output_dir ./hdf5_features \
    --split_mode test
	
# Split joint training set and package them
python data_preparation/package_features.py \
    --qfinder_dir ./extracted_features_joint_train_val/QFinder \
    --rhasfinder_dir ./extracted_features_joint_train_val/RHASFinder \
    --ffinder_dir ./extracted_features_joint_train_val/FFinder \
    --output_dir ./hdf5_features_joint \
    --split_mode random \
    --train_ratio 0.8
	
# Split tuning set and package them
python data_preparation/package_features.py \
    --qfinder_dir ./extracted_features_tuning_train_val/QFinder \
    --rhasfinder_dir ./extracted_features_tuning_train_val/RHASFinder \
    --ffinder_dir ./extracted_features_tuning_train_val/FFinder \
    --output_dir ./hdf5_features_tuning \
    --split_mode random \
    --train_ratio 0.8
	
# Package real HSSP test set (no splitting)
python data_preparation/package_features.py \
    --qfinder_dir ./extracted_features_hssp1471_test/QFinder \
    --rhasfinder_dir ./extracted_features_hssp1471_test/RHASFinder \
    --ffinder_dir ./extracted_features_hssp1471_test/FFinder \
    --output_dir ./hdf5_features_hssp1471 \
    --split_mode test
```

**Note**: Unlike Steps 2 and 3, Step 4 requires manually specifying input feature directories. All output HDF5 files are saved in a single directory, distinguished by filenames (e.g., `*_train_val.h5` and `*_test.h5`). See `data_preparation/package_features.py` for details.

#### Step 5: Train Models

```bash
python training/scripts/train_QFinder.py --config configs/QFinder_config.yaml

python training/scripts/train_RHASFinder.py --config configs/RHASFinder_config.yaml

python training/scripts/train_FFinder.py --config configs/FFinder_config.yaml
```

Training logs are automatically generated in `lightning_logs/`, which can be viewed using TensorBoard:

```bash
# View training logs for QFinder
tensorboard --logdir lightning_logs/QFinder
```
#### Step 6: Fine-Tuning Models
```bash
python tuning/tuning_QFinder.py --config configs/QFinder_config.yaml --pretrained_ckpt lightning_logs/QFinder/checkpoints/last.ckpt

python tuning/tuning_RHASFinder.py --config configs/RHASFinder_config.yaml --pretrained_ckpt lightning_logs/RHASFinder/checkpoints/last.ckpt

python tuning/tuning_FFinder.py --config configs/FFinder_config.yaml
```
### Step 7: Test Models

After training, the best model checkpoint path will be printed. Use that path for testing. Here, We uploaded pretrained model to test directly.

```bash
# Test QFinder
python testing/test_QFinder.py \
    --pretrained_model lightning_logs/QFinder/checkpoints/last.ckpt \
    --test_paths ./hdf5_features/QFinder_feature_test.h5 \
    --top_k 3

# Test RHASFinder
python testing/test_RHASFinder.py \
    --pretrained_model lightning_logs/RHASFinder/checkpoints/last.ckpt \
    --test_paths ./hdf5_features/RHASFinder_feature_test.h5 \
    --top_k 3

# Test FFinder
python testing/test_FFinder.py \
    --pretrained_model models/FFinder/FFinder_last.joblib \
    --test_paths ./hdf5_features/FFinder_feature_test.h5
```

Some metrices will be printed.

## Project Structure

```
ProtFinder/
├── data_preparation/          # Step 1-4: Data preparation scripts
│   ├── empirical_dist.py      # Step 1: Fit empirical distributions
│   ├── simulation.py          # Step 2: Generate simulated data
│   ├── feature_extraction.py  # Step 3: Extract features
│   └── package_features.py    # Step 4: Package features to HDF5
│
├── training/                  # Step 5: Model training
│   ├── modules/               # PyTorch Lightning modules
│   │   ├── QFinder_lightning.py
│   │   └── RHASFinder_lightning.py
│   └── scripts/               # Training scripts
│       ├── train_QFinder.py
│       ├── train_RHASFinder.py
│       └── train_FFinder.py
│
├── tuning/                   # Step 6: Fine-tuning files
│   ├── tuning_QFinder.py
│   ├── tuning_RHASFinder.py
│   └── tuning_FFinder.py
│
├── testing/                   # Step 7: Model testing
│   ├── test_QFinder.py
│   ├── test_RHASFinder.py
│   ├── test_FFinder.py
│   └── callbacks.py
│
├── models/                    # Model definitions
│   ├── QFinder.py
│   └── RHASFinder.py
│
├── data/                      # Data processing modules
│   └── datasets.py            # PyTorch Dataset classes
│
├── configs/                   # Configuration files
│   ├── QFinder_config.yaml
│   ├── RHASFinder_config.yaml
│   └── FFinder_config.yaml
│
├── empirical_parameters/      # Input CSV files for Step 1
├── fitted_empirical_dist/     # Output .npz files from Step 1
├── real_alignments_15330_train_val/     # 15330 real MSA data, extracted from zip files
├── hssp1471_test				  		 # 1471 real HSSP MSA data, extracted from zip files
├── pyproject.toml            			 # Project configuration
└── README.md
```

## Parameter Files

Parameter files in `empirical_parameters/` contain model parameters from the EvoNAPS database. They were computed by IQ-TREE.
- **F_parameters.csv**: Amino acid frequencies (20 columns: FREQ_A, FREQ_R, FREQ_N, FREQ_D, FREQ_C, FREQ_Q, FREQ_E, FREQ_G, FREQ_H, FREQ_I, FREQ_L, FREQ_K, FREQ_M, FREQ_F, FREQ_P, FREQ_S, FREQ_T, FREQ_W, FREQ_Y, FREQ_V)
- **G4_parameters.csv**: Gamma distribution parameters (1 column: G4)
- **I_parameters.csv**: Invariant site parameters (1 column: I)
- **external_lengths.csv**: External branch lengths (1 column: external)
- **internal_lengths.csv**: Internal branch lengths (1 column: internal)

## Models

### QFinder
- **Task**: 7-class substitution model classification
- **Classes**: LG, WAG, JTT, Q.plant, Q.bird, Q.mammal, Q.pfam
- **Architecture**: CNN with Squeeze-and-Excitation blocks
- **Input**: QFinder feature reshaped to (440, 25, 25)

### RHASFinder
- **Task**: 4-class RHAS model classification
- **Classes**: None, +G, +I, +G+I
- **Architecture**: Transformer encoder-based network
- **Input**: RHASFinder features

### FFinder
- **Task**: 2-class +F model classification
- **Classes**: -F, +F
- **Architecture**: XGBoost Classifier
- **Input**: FFinder feature

## Requirements

See `pyproject.toml` for dependencies. In addition, IQ-TREE is needed for simulation.
