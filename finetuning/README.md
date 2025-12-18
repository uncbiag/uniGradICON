# Finetuning uniGradICON on Your Data

This guide shows you how to finetune uniGradICON on your own datasets using configuration files. The finetuning system supports multiple datasets, weighted sampling, and segmentation-based training.

## 📋 Table of Contents
- [Quick Start](#-quick-start)
- [Step-by-Step Guide](#-step-by-step-guide)
- [Configuration Guide](#-configuration-guide)
- [Dataset Types](#-dataset-types)
- [Advanced Features](#-advanced-features)

## 🚀 Quick Start

```bash
# Navigate to finetuning directory
cd uniGradICON/
pip install .

cd uniGradICON/finetuning
# Run with example config
python finetune.py --config configs/config.yaml
```

## 📖 Step-by-Step Guide

### Step 1: Prepare Your Data

Organize your medical images in a directory structure. The system uses glob patterns to find images:

```
/path/to/data/
├── patient001.nii.gz
├── patient002.nii.gz
├── patient003.nii.gz
└── ...
```

For paired datasets (same patient, different timepoints):
```
/path/to/data/
├── patient001_t0.nii.gz
├── patient001_t1.nii.gz
├── patient002_t0.nii.gz
├── patient002_t1.nii.gz
└── ...
```

For segmentation-based training:
```
/path/to/images/
├── patient001.nii.gz
├── patient002.nii.gz
└── ...

/path/to/segmentations/
├── patient001_seg.nii.gz
├── patient002_seg.nii.gz
└── ...
```

### Step 2: Create a Configuration File

Create a YAML file (e.g., `my_config.yaml`) in the `configs/` directory:

```yaml
experiment_name: "my_finetuning_experiment"
model: "unigradicon"  # or "multigradicon"

training:
  batch_size: 4
  gpus: [0, 1]  # GPU device IDs
  epochs: 100
  eval_period: 10  # Validate every N epochs
  save_period: 50  # Save checkpoint every N epochs
  learning_rate: 0.00005
  input_shape: [175, 175, 175]  # Target image size
  weights_path: "unigradicon"  # Auto-downloads if not found
  output_folder: "results"
  
  # Loss configuration
  lmbda: 1.5  # Regularization weight
  similarity: "lncc"  # Options: "lncc", "lncc2", "mind"
  lncc_sigma: 5  # For LNCC losses

datasets:
  - name: "my_dataset"
    weight: 1.0  # Sampling weight (must sum to 1.0 across all datasets)
    type: "unpaired"  # See Dataset Types below
    image_glob: "/path/to/data/*.nii.gz"
    maximum_images: null  # Optional: limit number of images
    shuffle: true
    is_ct: false  # Set to true for CT images
    quantile_range: [0.01, 0.99]  # For MRI normalization
```

### Step 3: Start Training

```bash
python finetune.py --config configs/my_config.yaml
```

### Step 4: Monitor Training

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir=results/my_finetuning_experiment
```

### Step 5: Use Your Finetuned Model

After training, use your model weights for inference:

```bash
# Your weights are saved in results/[experiment_name]/checkpoints/
ls results/my_finetuning_experiment/checkpoints/

# Use with uniGradICON CLI
unigradicon-register \
  --fixed fixed.nii.gz \
  --moving moving.nii.gz \
  --transform_out transform.hdf5 \
  --warped_moving_out warped.nii.gz \
  --network_weights results/my_finetuning_experiment/checkpoints/network_weights_100
```

## ⚙️ Configuration Guide

### Training Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `batch_size` | int | Images per GPU | 4 |
| `gpus` | list | GPU device IDs | [0] |
| `epochs` | int | Training epochs | 500 |
| `learning_rate` | float | Adam learning rate | 5e-5 |
| `input_shape` | list | Target image dimensions [D,H,W] | [175,175,175] |
| `eval_period` | int | Validate every N epochs | 15 |
| `save_period` | int | Save checkpoint every N epochs | 50 |
| `lmbda` | float | Regularization weight | 1.5 |
| `similarity` | str | Loss function: "lncc", "lncc2", "mind" | "lncc" |
| `samples_per_epoch` | int | Samples per epoch (optional) | null |

### Dataset Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `name` | str | Dataset identifier | ✓ |
| `weight` | float | Sampling weight | ✓ |
| `type` | str | Dataset type (see below) | ✓ |
| `image_glob` | str | Glob pattern for images | ✓ |
| `match_regex` | str | Regex for pairing (paired types) | Conditional |
| `segmentation_glob` | str | Glob for segmentations (seg types) | Conditional |
| `seg_match_regex` | str | Regex for image-seg matching | Optional |
| `maximum_images` | int | Limit number of images | Optional |
| `use_cache` | bool | Enable/disable caching | true |
| `is_ct` | bool | CT vs MRI preprocessing | false |

## 🗂️ Dataset Types

### 1. Unpaired Dataset (`unpaired`)
Random pairs of images from different subjects.

```yaml
datasets:
  - name: "brain_mri"
    type: "unpaired"
    image_glob: "/data/brain/*.nii.gz"
    weight: 1.0
```

### 2. Paired Dataset (`paired`)
Matched pairs of images from the same subject (e.g., different timepoints).

```yaml
datasets:
  - name: "lung_followup"
    type: "paired"
    image_glob: "/data/lung/*_t*.nii.gz"
    match_regex: "patient(\\d+)_t"  # Groups by patient ID
    weight: 1.0
```

**Note:** The `match_regex` should capture a group that identifies the subject. Images with the same group ID will be paired together.

### 3. Unpaired with Segmentation (`unpaired_with_seg`)
Random pairs with segmentation guidance using Dice loss.

```yaml
datasets:
  - name: "brain_structures"
    type: "unpaired_with_seg"
    image_glob: "/data/images/*.nii.gz"
    segmentation_glob: "/data/segmentations/*_seg.nii.gz"
    seg_match_regex: "(\\d+)\\.nii"  # Extract subject ID
    weight: 1.0

training:
  dice_loss_weight: 0.3
```

**Note:** Only images with matching segmentations will be loaded. The system automatically filters images.

### 4. Paired with Segmentation (`paired_with_seg`)
Paired images with segmentation guidance.

```yaml
datasets:
  - name: "cardiac_phases"
    type: "paired_with_seg"
    image_glob: "/data/cardiac/*_phase*.nii.gz"
    segmentation_glob: "/data/cardiac/*_seg.nii.gz"
    match_regex: "patient(\\d+)_phase"  # For pairing images
    seg_match_regex: "patient(\\d+)_"   # For matching segmentations
    weight: 1.0

training:
  dice_loss_weight: 0.3
```

## 🔬 Advanced Features

### Multi-Dataset Training

Train on multiple datasets simultaneously with weighted sampling:

```yaml
datasets:
  - name: "brain_t1"
    weight: 0.4  # 40% of samples from this dataset
    type: "unpaired"
    image_glob: "/data/brain_t1/*.nii.gz"
  
  - name: "brain_t2"
    weight: 0.3  # 30% of samples
    type: "unpaired"
    image_glob: "/data/brain_t2/*.nii.gz"
  
  - name: "lung_ct"
    weight: 0.3  # 30% of samples
    type: "unpaired"
    image_glob: "/data/lung/*.nii.gz"
    is_ct: true
    ct_window: [-1000, 1000]
```

**Note:** Weights must sum to 1.0.

You can find examples in the `configs/` directory.

### Auto-Download Pretrained Weights

Specify model name instead of path to auto-download:

```yaml
training:
  weights_path: "unigradicon"  # Auto-downloads uniGradICON weights
  # OR
  weights_path: "multigradicon"  # Auto-downloads multiGradICON weights
  # OR
  weights_path: "/path/to/my/weights.trch"  # Use custom weights
```

### Resume Training

The system automatically detects if you're resuming from a checkpoint:

```yaml
training:
  weights_path: "results/my_experiment/checkpoints/network_weights_50"
```

If `optimizer_weights_50` exists, training resumes with optimizer state. Otherwise, it starts fresh with the model weights.

### Control Samples Per Epoch

For large datasets or faster testing:

```yaml
training:
  samples_per_epoch: 4000  # Process 4000 samples per epoch
```

Without this parameter, all dataset samples are used each epoch.

### Disable Caching

For debugging or frequently changing data:

```yaml
datasets:
  - name: "test_dataset"
    type: "unpaired"
    image_glob: "/data/*.nii.gz"
    use_cache: false  # Reload images every time
```

**Default:** Caching is enabled. Cached data is stored in `results/[dataset_name]_cache/`.

### CT vs MRI Preprocessing

**MRI (default):**
```yaml
datasets:
  - name: "mri_dataset"
    is_ct: false
    quantile_range: [0.01, 0.99]  # Normalize using quantiles
```

**CT:**
```yaml
datasets:
  - name: "ct_dataset"
    is_ct: true
    ct_window: [-1000, 1000]  # HU windowing
```

## 🐛 Troubleshooting

### "No images found"
- Check your `image_glob` pattern is correct
- Use absolute paths, not relative paths
- Test with: `ls /your/path/*.nii.gz`

### "No segmentations found" or "Filtered to 0/N images"
- Verify `seg_match_regex` correctly extracts subject IDs from image paths
- Check that segmentation files exist for your images
- Print extracted IDs: `python -c "import re; print(re.search('your_regex', 'your_path').group(1))"`

### "Weights must sum to 1.0"
- Check all dataset `weight` values sum to 1.0
- Example: [0.5, 0.3, 0.2] ✓  |  [0.5, 0.4, 0.3] ✗

### Cache takes too much disk space
- Set `use_cache: false`
- Delete old caches: `rm -rf results/*_cache`
- Use `maximum_images` to limit dataset size


