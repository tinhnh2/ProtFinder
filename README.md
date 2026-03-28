# Phylogenetic Model Selection via Machine Learning

This project implements machine learning models for phylogenetic model selection.

## Project Overview

The project consists of 5 main steps:

1. **Empirical Distribution Fitting** - Fit inverse CDF distributions from real MSA parameters
2. **Data Simulation** - Generate simulated MSAs using IQ-TREE Ali-Sim
3. **Feature Extraction** - Extract features from simulated MSAs
4. **Feature Packaging** - Package features into HDF5 format
5. **Model Training and Testing** - Train and evaluate machine learning models


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
    --iqtree_path /usr/bin/iqtree3 \
    --param_dir fitted_empirical_dist \
    --trees_dir ./simulated_trees \
    --output_dir ./simulated_alignments \
    --data_type train_val \
    --num_iterations 200

# Generate test set
python data_preparation/simulation.py \
    --iqtree_path /usr/bin/iqtree3 \
    --param_dir fitted_empirical_dist \
    --trees_dir ./simulated_trees \
    --output_dir ./simulated_alignments \
    --data_type test \
    --num_iterations 40
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
    --rasfinder_dir ./extracted_features_train_val/RASFinder \
    --ffinder_dir ./extracted_features_train_val/FFinder \
    --output_dir ./hdf5_features \
    --split_mode random \
    --train_ratio 0.8

# Package test set (no splitting)
python data_preparation/package_features.py \
    --qfinder_dir ./extracted_features_test/QFinder \
    --rasfinder_dir ./extracted_features_test/RASFinder \
    --ffinder_dir ./extracted_features_test/FFinder \
    --output_dir ./hdf5_features \
    --split_mode test
```

**Note**: Unlike Steps 2 and 3, Step 4 requires manually specifying input feature directories. All output HDF5 files are saved in a single directory, distinguished by filenames (e.g., `*_train_val.h5` and `*_test.h5`). See `data_preparation/package_features.py` for details.

#### Step 5: Train Models

```bash
python training/scripts/train_QFinder.py --config configs/QFinder_config.yaml

python training/scripts/train_RASFinder.py --config configs/RASFinder_config.yaml

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

python tuning/tuning_RASFinder.py --config configs/RASFinder_config.yaml --pretrained_ckpt lightning_logs/RASFinder/checkpoints/last.ckpt

python tuning/tuning_FFinder.py --config configs/FFinder_config.yaml
```
### Step 7: Test Models

After training, the best model checkpoint path will be printed. Use that path for testing.

```bash
# Test QFinder
python testing/test_QFinder.py \
    --checkpoint lightning_logs/QFinder/checkpoints/QFinder-epoch=XX-val_acc=X.XXXX.ckpt \
    --test_h5_paths ./hdf5_features/QFinder_feature_test.h5

# Test RASFinder
python testing/test_RASFinder.py \
    --checkpoint lightning_logs/RASFinder/checkpoints/RASFinder-epoch=XX-val_acc=X.XXXX.ckpt \
    --test_h5_paths ./hdf5_features/RASFinder_feature_test.h5

# Test FFinder
python testing/test_FFinder.py \
    --checkpoint models/FFinder/FFinder_last.joblib \
    --test_h5_paths ./hdf5_features/FFinder_feature_test.h5
```

Some metrices will be printed.

## Project Structure

```
model-selection-via-ML/
├── data_preparation/          # Step 1-4: Data preparation scripts
│   ├── empirical_dist.py      # Step 1: Fit empirical distributions
│   ├── simulation.py          # Step 2: Generate simulated data
│   ├── feature_extraction.py  # Step 3: Extract features
│   └── package_features.py    # Step 4: Package features to HDF5
│
├── training/                  # Step 5: Model training
│   ├── modules/               # PyTorch Lightning modules
│   │   ├── QFinder_lightning.py
│   │   └── RASFinder_lightning.py
│   └── scripts/               # Training scripts
│       ├── train_QFinder.py
│       ├── train_RASFinder.py
│       └── train_FFinder.py
│
├── testing/                   # Step 6: Model testing
│   ├── test_QFinder.py
│   ├── test_RASFinder.py
│   ├── test_FFinder.py
│   └── callbacks.py
│
├── models/                    # Model definitions
│   ├── QFinder.py
│   └── RASFinder.py
│
├── data/                      # Data processing modules
│   └── datasets.py            # PyTorch Dataset classes
│
├── configs/                   # Configuration files
│   ├── QFinder_config.yaml
│   ├── RASFinder_config.yaml
│   └── FFinder_config.yaml
|
├── tuning/                   # Fine-tuning files
│   ├── tuning_QFinder.py
│   ├── tuning_RASFinder.py
│   └── tuning_FFinder.py
│
├── empirical_parameters/      # Input CSV files for Step 1
├── fitted_empirical_dist/     # Output .npz files from Step 1
├── real_alignments_15330/     # 15330 real MSA data, need extracted from zip files
├── hssp1471				   # 1471 real HSSP MSA data, need extracted from zip files
├── pyproject.toml             # Project configuration
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

### RASFinder
- **Task**: 4-class RHAS model classification
- **Classes**: None, +G, +I, +G+I
- **Architecture**: Transformer encoder-based network
- **Input**: RASFinder features

### FFinder
- **Task**: 2-class +F model classification
- **Classes**: -F, +F
- **Architecture**: Balanced Random Forest Classifier
- **Input**: FFinder feature

## Requirements

See `pyproject.toml` for dependencies. In addition, IQ-TREE is needed for simulation.
