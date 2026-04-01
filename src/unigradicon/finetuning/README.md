# Finetuning uniGradICON on Your Data

This guide shows you how to finetune uniGradICON on your own datasets using configuration files. The finetuning system supports multiple datasets, weighted sampling, and segmentation-based training.

## Table of Contents
- [Quick Start](#quick-start)
- [Example: Finetuning on Learn2Reg Data](#example-finetuning-on-learn2reg-data)
- [Step-by-Step Guide](#step-by-step-guide)
- [Configuration Guide](#configuration-guide)
- [Dataset Types](#dataset-types)
- [Advanced Features](#advanced-features)
- [Dice Loss and Masking](#dice-loss-and-masking)

## Quick Start

**Requirements:** GPU required (CUDA).

**Install (PyPI or source):**
- PyPI: `pip install unigradicon`
- Dev/source: `pip install -e .` from the repo root

```bash
# Run with your config
unigradicon-finetune --config /path/to/your_config.yaml

# Example with repo checkout (after pip install -e .)
unigradicon-finetune --config configs/examples/config.yaml
```

## Example: Finetuning on Learn2Reg Data

This section walks through finetuning uniGradICON on three public [Learn2Reg](https://learn2reg.grand-challenge.org/) datasets. Ready-to-use configs are provided for each dataset individually and for multi-dataset training.

| Config | Dataset | Type | Modality | Similarity | Pretrained | Images |
|--------|---------|------|----------|------------|------------|--------|
| `l2r_oasis.yaml` | OASIS brain MRI | `unpaired_with_seg` | MRI | lncc | uniGradICON | 414 |
| `l2r_lungct.yaml` | LungCT | `paired_with_seg` | CT | lncc | uniGradICON | 40 (20×2) |
| `l2r_abdomenmrct.yaml` | AbdomenMRCT | `unpaired_with_seg` | CT + MR | lncc2 | multiGradICON | 105 (48 CT + 57 MR) |
| `l2r_multi.yaml` | All three combined | mixed | MRI + CT | lncc2 | multiGradICON | 559 |

Cross-modality datasets (AbdomenMRCT) use per-image `"modality"` fields in the JSON so each image is preprocessed according to its own modality (CT windowing vs MRI quantile normalization). `lncc2` (SquaredLNCC) is used as a modality-invariant similarity measure, and `multigradicon` provides pretrained weights for multimodal registration.

### 1. Install uniGradICON

```bash
pip install unigradicon
# or from source: pip install -e .
```

### 2. Download the datasets

Download the **training splits** from the [Learn2Reg Datasets page](https://learn2reg.grand-challenge.org/Datasets/) (requires a free Grand Challenge account):

- **OASIS** — brain MRI with 35 anatomical structure labels
- **LungCT** — paired inspiration/expiration lung CT with lung masks
- **AbdomenMRCT** — abdomen CT/MR with organ labels (cross-modality)

### 3. Extract into a `datasets/` directory

```bash
mkdir -p datasets
# Move downloaded zip files into datasets/ then extract:
cd datasets
unzip OASIS.zip
unzip LungCT.zip
unzip AbdomenMRCT.zip
cd ..
```

Your directory should look like:
```
datasets/
├── OASIS/
│   ├── imagesTr/       # 414 brain MRI scans
│   └── labelsTr/       # 35-structure segmentation labels
├── LungCT/
│   ├── imagesTr/       # 40 lung CT scans (20 subjects × 2 timepoints)
│   └── masksTr/        # lung masks
└── AbdomenMRCT/
    ├── imagesTr/       # 105 images (48 CT + 57 MR, cross-modality)
    └── labelsTr/       # organ labels
```

### 4. Generate JSON dataset files

```bash
python scripts/prepare_l2r_datasets.py --data_dir datasets/
```

This scans the extracted data and creates JSON files (`l2r_oasis.json`, `l2r_lungct.json`, `l2r_abdomenmrct.json`) in `configs/learn2reg/`. You only need to run this once.

### 5. Start finetuning

Pick any of the provided configs:

```bash
# Brain MRI (unpaired, 414 images, ~3 MB each)
unigradicon-finetune --config configs/learn2reg/l2r_oasis.yaml

# Lung CT (paired, 20 subjects × 2 timepoints)
unigradicon-finetune --config configs/learn2reg/l2r_lungct.yaml

# Abdomen MRCT (105 CT+MR images with per-image modality, lncc2 + multiGradICON)
unigradicon-finetune --config configs/learn2reg/l2r_abdomenmrct.yaml

# Multi-dataset (all three, weighted sampling)
unigradicon-finetune --config configs/learn2reg/l2r_multi.yaml
```

### 6. Monitor and use results

```bash
# Monitor training
tensorboard --logdir results/<experiment_name>/logs

# Use finetuned weights for inference
unigradicon-register \
  --fixed fixed.nii.gz --moving moving.nii.gz \
  --fixed_modality mri --moving_modality mri \
  --quantile_range 0.01 0.99 \
  --transform_out transform.hdf5 \
  --network_weights results/l2r_oasis_brain_mri/checkpoints/network_weights_final.trch
```

## Step-by-Step Guide

### Step 1: Prepare Your Data

Organize your data and create a JSON file to define your datasets. All datasets use a consistent JSON format with a `data` list.

**Example `dataset.json` for unpaired data:**
```json
{
  "data": [
    {"image": "/path/to/img1.nii.gz"},
    {"image": "/path/to/img2.nii.gz"}
  ]
}
```

**Note:** All datasets require at least 2 images. Paired datasets need at least 2 images per subject. For mixed-modality datasets, add `"modality": "ct"` or `"modality": "mri"` per entry (see [Mixed Modality Datasets](#mixed-modality-datasets)).

**Example `dataset.json` for paired data:**
```json
{
  "data": [
    {"image": "/path/p1_t0.nii.gz", "subject_id": "p1"},
    {"image": "/path/p1_t1.nii.gz", "subject_id": "p1"}
  ]
}
```

### Step 2: Create a Configuration File

Create a YAML file (e.g., `my_config.yaml`) in the `configs/` directory:

```yaml
experiment:
  name: "my_finetuning_experiment"
  model_weights: "unigradicon"  # Auto-downloads if not found

training:
  batch_size: 4
  gpus: [0, 1]  # GPU device IDs
  epochs: 100
  eval_period: 10  # Validate every N epochs
  save_period: 50  # Save checkpoint every N epochs
  learning_rate: 0.00005
  input_shape: [175, 175, 175]  # Target image size
  seed: 42  # Optional: for reproducibility

  # Loss configuration
  lambda: 1.5  # Regularization weight
  similarity: "lncc"  # Options: "lncc", "lncc2", "mind"
  lncc_sigma: 5  # For LNCC losses
  dice_loss_weight: 0.0  # >0 only for segmentation datasets
  loss_function_masking: false  # When true, Dice is disabled and must stay 0.0

datasets:
  - name: "my_dataset"
    weight: 1.0  # Relative sampling weight (any positive value)
    type: "unpaired"  # See Dataset Types below
    json_file: "my_dataset.json"
    maximum_images: null  # Optional: limit number of images
    shuffle: true
    is_ct: false  # Set to true for CT images
    quantile_range: [0.01, 0.99]  # For MRI normalization
```

Values shown above are examples; see [Training Parameters](#training-parameters) for defaults.

### Step 3: Start Training

```bash
unigradicon-finetune --config configs/my_config.yaml
```

### Step 4: Monitor Training

Training progress is logged to TensorBoard. Validation writes scalar losses plus image panels
(moving/fixed/warped/difference), and segmentation panels when segmentation datasets are used:

```bash
# Footsteps stores runs in results/<experiment.name>/logs/<timestamp>
tensorboard --logdir="results/my_finetuning_experiment/logs"
# If you rerun with the same name, use the suffixed folder (e.g., results/my_finetuning_experiment-1/logs)
```

### Step 5: Use Your Finetuned Model

After training, use your model weights for inference:

```bash
# Your weights are saved in results/<experiment.name>/checkpoints/
ls results/my_finetuning_experiment/checkpoints/

# Use with uniGradICON CLI
unigradicon-register \
  --fixed fixed.nii.gz \
  --moving moving.nii.gz \
  --fixed_modality mri --moving_modality mri \
  --transform_out transform.hdf5 \
  --warped_moving_out warped.nii.gz \
  --network_weights results/my_finetuning_experiment/checkpoints/network_weights_final.trch
```

### Matching Preprocessing Between Finetuning and Inference

Finetuning and inference use separate preprocessing pipelines. By default they differ
slightly for MRI: finetuning clips intensities at both quantile bounds (default `[0.01, 0.99]`),
while inference uses the actual image minimum and the 99th percentile.

To ensure consistent behavior, pass the same preprocessing parameters at inference time
using `--quantile_range` (for MRI) or `--ct_window` (for CT):

```bash
# MRI: match finetuning's default quantile_range of [0.01, 0.99]
unigradicon-register \
  --fixed fixed.nii.gz --moving moving.nii.gz \
  --fixed_modality mri --moving_modality mri \
  --quantile_range 0.01 0.99 \
  --transform_out transform.hdf5 \
  --network_weights results/my_experiment/checkpoints/network_weights_final.trch

# CT: match a custom ct_window used during finetuning
unigradicon-register \
  --fixed fixed.nii.gz --moving moving.nii.gz \
  --fixed_modality ct --moving_modality ct \
  --ct_window -500 1500 \
  --transform_out transform.hdf5 \
  --network_weights results/my_experiment/checkpoints/network_weights_final.trch
```

If you used the default `ct_window: [-1000, 1000]` during finetuning, no extra flag is needed for CT — the inference default already matches.

## Configuration Guide

### Training Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `batch_size` | int | Images per GPU | 4 |
| `gpus` | list | GPU device IDs | [0] |
| `epochs` | int | Training epochs | 500 |
| `learning_rate` | float | Adam learning rate | 5e-5 |
| `input_shape` | list | Target image dimensions [D,H,W] used during finetuning | [175,175,175] |
| `eval_period` | int | Validate every N epochs | 15 |
| `save_period` | int | Save checkpoint every N epochs | 50 |
| `seed` | int | Random seed for reproducibility | null |
| `lambda` | float | Regularization weight | 1.5 |
| `similarity` | str | Loss function: "lncc", "lncc2", "mind" | "lncc" |
| `dice_loss_weight` | float | Dice loss term weight for segmentation mode | 0.0 |
| `loss_function_masking` | bool | Apply segmentation mask to similarity loss (segmentation mode only) | false |
| `lncc_sigma` | int | Sigma for LNCC / SquaredLNCC similarity | 5 |
| `mind_radius` | int | Radius for MIND-SSC similarity | 2 |
| `mind_dilation` | int | Dilation for MIND-SSC similarity | 2 |
| `samples_per_epoch` | int | Samples per epoch (optional) | null |
| `num_workers` | int | DataLoader worker processes | 4 |

### Input Shape Guidance

- The released `unigradicon` / `multigradicon` weights were trained at `input_shape: [175, 175, 175]`.
- You can finetune with a different `input_shape`, but convergence may be slower and you may need more epochs.
- For the most stable transfer behavior, keep finetuning and inference preprocessing shapes consistent.
- If you finetune at a very different shape, inference quality can change because the model sees a different resampling distribution than pretraining.

### Dataset Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `name` | str | Dataset identifier | Required |
| `type` | str | Dataset type (see below) | Required |
| `json_file` | str | Path to JSON dataset definition | Required |
| `weight` | float | Relative sampling weight | 1.0 |
| `maximum_images` | int | Limit number of images | null |
| `use_cache` | bool | Enable/disable caching | true |
| `cache_dir` | str | Directory for cached datasets | null |
| `read_type` | str | Image reader: `"itk"` (NIfTI/NRRD) or `"dicom"` (DICOM series directories) | `"itk"` |
| `shuffle` | bool | Shuffle image order before loading | true |
| `is_ct` | bool | CT vs MRI preprocessing | false |
| `ct_window` | list | HU window for CT [min, max] | [-1000, 1000] |
| `quantile_range` | list | Intensity quantile range for MRI | [0.01, 0.99] |

`json_file` paths are resolved relative to the YAML config file's directory, so you can usually reference just the filename.

## Dataset Types

### 1. Unpaired Dataset (`unpaired`)
Random pairs of images from different subjects.

```yaml
datasets:
  - name: "brain_mri"
    type: "unpaired"
    json_file: "brain_mri.json"
    weight: 1.0
```

**JSON Format:**
```json
{
  "data": [
    {"image": "/path/to/img1.nii.gz"},
    {"image": "/path/to/img2.nii.gz"}
  ]
}
```

### 2. Paired Dataset (`paired`)
Matched pairs of images from the same subject.

```yaml
datasets:
  - name: "lung_followup"
    type: "paired"
    json_file: "lung_pairs.json"
    weight: 1.0
```

**JSON Format:**
```json
{
  "data": [
    {"image": "/path/p1_t0.nii.gz", "subject_id": "p1"},
    {"image": "/path/p1_t1.nii.gz", "subject_id": "p1"}
  ]
}
```

### 3. Unpaired with Segmentation (`unpaired_with_seg`)
Random pairs with segmentation guidance (Dice loss or loss masking).

```yaml
datasets:
  - name: "brain_structures"
    type: "unpaired_with_seg"
    json_file: "brain_seg.json"
    weight: 1.0
```

**JSON Format:**
```json
{
  "data": [
    {"image": "/path/img1.nii.gz", "segmentation": "/path/seg1.nii.gz"},
    {"image": "/path/img2.nii.gz", "segmentation": "/path/seg2.nii.gz"}
  ]
}
```

### 4. Paired with Segmentation (`paired_with_seg`)
Paired images with segmentation guidance.

```yaml
datasets:
  - name: "cardiac_phases"
    type: "paired_with_seg"
    json_file: "cardiac.json"
    weight: 1.0
```

**JSON Format:**
```json
{
  "data": [
    {"image": "/path/p1_t0.nii.gz", "segmentation": "/path/p1_t0_seg.nii.gz", "subject_id": "p1"},
    {"image": "/path/p1_t1.nii.gz", "segmentation": "/path/p1_t1_seg.nii.gz", "subject_id": "p1"}
  ]
}
```

## Advanced Features

### Multi-Dataset Training

Train on multiple datasets simultaneously with weighted sampling:

```yaml
datasets:
  - name: "brain_t1"
    weight: 0.4
    type: "unpaired"
    json_file: "brain_t1.json"
  
  - name: "brain_t2"
    weight: 0.3
    type: "unpaired"
    json_file: "brain_t2.json"
  
  - name: "lung_ct"
    weight: 0.3
    type: "unpaired"
    json_file: "lung_ct.json"
    is_ct: true
    ct_window: [-1000, 1000]
```

**Note:** Weights are relative; they do not need to sum to 1.0.

## Dice Loss and Masking

When training with segmentation datasets (`unpaired_with_seg` or `paired_with_seg`), the total loss is:

`L_total = lambda * L_inverse_consistency + L_similarity + dice_loss_weight * L_dice`

- `dice_loss_weight` controls how strongly segmentation overlap is optimized.
- Set `dice_loss_weight: 0.0` to disable Dice.

### Important masking rule

If you enable:

```yaml
training:
  loss_function_masking: true
```

then Dice loss is not calculated in finetuning. In this mode, `dice_loss_weight` must be `0.0`.

When `loss_function_masking: true`, the segmentation mask is passed directly to the similarity loss function (e.g., LNCC), restricting the loss computation to regions where the segmentation is present. This is useful when you want the registration to focus on a specific anatomical region without adding a separate Dice loss term.

**Choosing between Dice loss and loss masking:**
- Use `dice_loss_weight > 0` when you want segmentation overlap as an explicit optimization target alongside the image similarity loss.
- Use `loss_function_masking: true` when you want to restrict the similarity loss to a region of interest defined by the segmentation, without optimizing segmentation overlap directly.
- These two modes are mutually exclusive in finetuning.

### Auto-Download Pretrained Weights

Specify model name instead of path to auto-download:

```yaml
experiment:
  model_weights: "unigradicon"      # Auto-downloads uniGradICON weights
  # OR
  model_weights: "multigradicon"    # Auto-downloads multiGradICON weights
  # OR
  model_weights: "/path/to/my/weights.trch"  # Use custom weights
```

### Resume Training

The system automatically detects if you're resuming from a checkpoint:

```yaml
experiment:
  model_weights: "results/my_experiment/checkpoints/network_weights_50.trch"
```

If `optimizer_weights_50.trch` exists, training resumes with optimizer state. Otherwise, it starts fresh with the model weights.

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
    json_file: "test.json"
    use_cache: false  # Reload images every time
```

**Default:** Caching is enabled. Cached data is stored in the experiment's output directory, or in `cache_dir` if specified in the dataset config.

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

To use the same preprocessing at inference time, see [Matching Preprocessing Between Finetuning and Inference](#matching-preprocessing-between-finetuning-and-inference).

### Mixed Modality Datasets

When a dataset contains both CT and MRI images (e.g., cross-modality registration), add a `modality` field to each entry in the JSON file. Each image is then preprocessed according to its own modality, regardless of the dataset-level `is_ct` setting:

```json
{
  "data": [
    {"image": "/path/patient1_mr.nii.gz", "modality": "mri", "subject_id": "p1"},
    {"image": "/path/patient1_ct.nii.gz", "modality": "ct", "subject_id": "p1"},
    {"image": "/path/patient2_mr.nii.gz", "modality": "mri", "subject_id": "p2"},
    {"image": "/path/patient2_ct.nii.gz", "modality": "ct", "subject_id": "p2"}
  ]
}
```

MRI images use quantile normalization (`quantile_range`), CT images use HU windowing (`ct_window`). Both parameters can be set in the dataset config:

```yaml
datasets:
  - name: "abdomen_mrct"
    type: "paired"
    json_file: "mrct_data.json"
    weight: 1.0
    quantile_range: [0.01, 0.99]  # Applied to MRI images
    ct_window: [-1000, 1000]      # Applied to CT images
```

If `modality` is not specified for an entry, the dataset-level `is_ct` setting is used as fallback.

## Troubleshooting

### "JSON file not found"
- Check that your `json_file` path is correct and accessible.
- Use absolute paths to avoid confusion.

### "Data must be provided"
- Ensure your JSON file contains the top-level `data` key.

### Weights are relative
- Sampler treats weights as relative multipliers; they do not need to sum to 1.0.
- Keep weights positive to avoid invalid sampler behavior.

### Cache takes too much disk space
- Set `use_cache: false`
- Delete old caches: `rm results/*/*_cached_*.trch`
- Use `maximum_images` to limit dataset size
