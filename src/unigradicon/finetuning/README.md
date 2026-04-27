# Finetuning uniGradICON on Your Data

This guide shows you how to finetune uniGradICON on your own datasets using configuration files. The finetuning system supports multiple datasets, weighted sampling, and segmentation/mask-based training.

## Table of Contents
- [Quick Start](#quick-start)
- [Example: Finetuning on Learn2Reg Data](#example-finetuning-on-learn2reg-data)
- [Step-by-Step Guide](#step-by-step-guide)
- [Configuration Guide](#configuration-guide)
- [Dataset Types](#dataset-types)
- [JSON Data Fields](#json-data-fields)
- [Segmentation, Masking, and Dice Loss](#segmentation-masking-and-dice-loss)
- [Label Randomization](#label-randomization-use_label)
- [Advanced Features](#advanced-features)

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
| `l2r_oasis.yaml` | OASIS brain MRI | `unpaired` | MRI | lncc | uniGradICON | 414 |
| `l2r_lungct.yaml` | LungCT | `paired` | CT | lncc | uniGradICON | 40 (20x2) |
| `l2r_abdomenmrct.yaml` | AbdomenMRCT | `unpaired` | CT + MR | lncc2 | multiGradICON | 105 (48 CT + 57 MR) |
| `l2r_multi.yaml` | All three combined | mixed | MRI + CT | lncc2 | multiGradICON | 559 |

Cross-modality datasets (AbdomenMRCT) use per-image `"modality"` fields in the JSON so each image is preprocessed according to its own modality (CT windowing vs MRI quantile normalization). `lncc2` (SquaredLNCC) is used as a modality-invariant similarity measure, and `multigradicon` provides pretrained weights for multimodal registration.

### 1. Install uniGradICON

```bash
pip install unigradicon
# or from source: pip install -e .
```

### 2. Download the datasets

Download the **training splits** from the [Learn2Reg Datasets page](https://learn2reg.grand-challenge.org/Datasets/) (requires a free Grand Challenge account):

- **OASIS**: brain MRI with 35 anatomical structure labels
- **LungCT**: paired inspiration/expiration lung CT with lung masks
- **AbdomenMRCT**: abdomen CT/MR with organ labels (cross-modality)

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
│   ├── imagesTr/       # 40 lung CT scans (20 subjects x 2 timepoints)
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
# Brain MRI (unpaired, 414 images)
unigradicon-finetune --config configs/learn2reg/l2r_oasis.yaml

# Lung CT (paired, 20 subjects x 2 timepoints)
unigradicon-finetune --config configs/learn2reg/l2r_lungct.yaml

# Abdomen MRCT (105 CT+MR images with per-image modality, lncc2 + multiGradICON)
unigradicon-finetune --config configs/learn2reg/l2r_abdomenmrct.yaml

# Multi-dataset (all three, weighted sampling)
unigradicon-finetune --config configs/learn2reg/l2r_multi.yaml
```

### 6. Monitor and use results

```bash
# Monitor training
tensorboard --logdir results/

# Use finetuned weights for inference
unigradicon-register \
  --fixed fixed.nii.gz --moving moving.nii.gz \
  --fixed_modality mri --moving_modality mri \
  --quantile_range 0.0 0.99 \
  --transform_out transform.hdf5 \
  --network_weights results/l2r_oasis_brain_mri/checkpoints/network_weights_final.trch
```

## Step-by-Step Guide

### Step 1: Prepare Your Data

Organize your data and create a JSON file. The JSON format uses a `data` list where each entry has an `image` path and optional fields for segmentations and masks:

```json
{
  "data": [
    {"image": "img1.nii.gz"},
    {"image": "img2.nii.gz"}
  ]
}
```

All datasets require at least 2 images. Paired datasets need at least 2 images per subject. Paths can be absolute or relative to the JSON file's directory. See [JSON Data Fields](#json-data-fields) for all supported fields.

### Step 2: Create a Configuration File

Create a YAML file (e.g., `my_config.yaml`):

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
  learning_rate: 0.00005  # Original pretraining LR
  input_shape: [175, 175, 175]  # Model input size (images are resampled to this)
  seed: 42  # Optional: for reproducibility

  # Loss configuration
  lambda: 1.5  # Regularization weight
  similarity: "lncc"  # Options: "lncc", "lncc2", "mind"
  lncc_sigma: 5  # For LNCC losses

datasets:
  - name: "my_dataset"
    weight: 1.0  # Relative sampling weight (any positive value)
    type: "unpaired"  # "unpaired" or "paired"
    json_file: "my_dataset.json"
    is_ct: false  # Set to true for CT images
    quantile_range: [0.0, 0.99]  # For MRI normalization
```

Values shown above are examples; see [Training Parameters](#training-parameters) for defaults.

### Step 3: Start Training

```bash
unigradicon-finetune --config configs/my_config.yaml
```

### Step 4: Monitor Training

Training progress is logged to TensorBoard. Validation writes scalar losses plus image panels
(moving/fixed/warped/difference), and segmentation panels when segmentation data is available:

```bash
# Footsteps stores runs in results/<experiment.name>/logs/<timestamp>
tensorboard --logdir="results/"
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

The default preprocessing parameters match between finetuning and inference, so no extra flags are needed if you use the defaults. If you customize `quantile_range` or `ct_window` in your finetuning config, pass the same values at inference time.

**Note:** The finetuning `modality` field accepts any string (e.g., `"t1"`, `"flair"`) where only `"ct"` triggers CT preprocessing. The CLI `--fixed_modality` / `--moving_modality` flags accept `"ct"` or `"mri"` only. Use `--fixed_modality mri` for any non-CT modality at inference.

```bash
# Example: custom ct_window used during finetuning
unigradicon-register \
  --fixed fixed.nii.gz --moving moving.nii.gz \
  --fixed_modality ct --moving_modality ct \
  --ct_window -500 1500 \
  --transform_out transform.hdf5 \
  --network_weights results/my_experiment/checkpoints/network_weights_final.trch
```

## Configuration Guide

### Training Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `batch_size` | int | Images per GPU | 4 |
| `gpus` | list | GPU device IDs | [0] |
| `epochs` | int | Training epochs | 500 |
| `learning_rate` | float | Adam learning rate | 5e-5 |
| `input_shape` | list | Model input dimensions [D,H,W] (images are resampled to this) | [175,175,175] |
| `eval_period` | int | Validate every N epochs | 10 |
| `save_period` | int | Save checkpoint every N epochs | 50 |
| `seed` | int | Random seed for reproducibility | null |
| `lambda` | float | Regularization weight | 1.5 |
| `similarity` | str | Loss function: "lncc", "lncc2", "mind" | "lncc" |
| `dice_loss_weight` | float | Dice loss weight (requires `segmentation` in JSON) | 0.0 |
| `loss_function_masking` | bool | Restrict similarity loss to masked regions (requires `mask` in JSON) | false |
| `roi_masking` | bool | Crop images to ROI before registration (requires `mask` in JSON) | false |
| `use_label` | bool | Label randomization for modality-invariant training (see below) | false |
| `lncc_sigma` | int | Sigma for LNCC / SquaredLNCC similarity | 5 |
| `mind_radius` | int | Radius for MIND-SSC similarity | 2 |
| `mind_dilation` | int | Dilation for MIND-SSC similarity | 2 |
| `samples_per_epoch` | int | Samples per epoch (optional) | total dataset size |
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
| `type` | str | `"unpaired"` or `"paired"` | Required |
| `json_file` | str | Path to JSON dataset definition | Required |
| `weight` | float | Relative sampling weight | 1.0 |
| `maximum_images` | int | Limit number of images | null |
| `use_cache` | bool | Enable/disable caching | true |
| `cache_dir` | str | Directory for cached datasets | null |
| `read_type` | str | Image reader: `"itk"` (NIfTI/NRRD) or `"dicom"` (DICOM series directories) | `"itk"` |
| `shuffle` | bool | Shuffle image order before loading | true |
| `is_ct` | bool | CT vs MRI preprocessing | false |
| `ct_window` | list | HU window for CT [min, max] | [-1000, 1000] |
| `quantile_range` | list | Intensity quantile range for MRI | [0.0, 0.99] |

`json_file` paths are resolved relative to the YAML config file's directory, so you can usually reference just the filename.

## Dataset Types

There are two dataset types, which control how image pairs are formed:

### Unpaired (`unpaired`)
Random pairs of images from the dataset.

```yaml
datasets:
  - name: "brain_mri"
    type: "unpaired"
    json_file: "brain_mri.json"
    weight: 1.0
```

### Paired (`paired`)
Matched pairs of images from the same subject (requires `subject_id` in JSON entries).

```yaml
datasets:
  - name: "lung_followup"
    type: "paired"
    json_file: "lung_pairs.json"
    weight: 1.0
```

## JSON Data Fields

Each JSON dataset file has a `data` list where each entry contains an `image` path and optional fields. The training configuration determines which optional fields are required and loaded:

- `dice_loss_weight > 0` requires `segmentation`
- `loss_function_masking: true` or `roi_masking: true` requires `mask`

Optional fields that are present in JSON but not required by the current training configuration are ignored.

| Field | Required | Description |
|-------|----------|-------------|
| `image` | Yes | Path to the image file |
| `segmentation` | No | Path to integer label map for Dice loss |
| `mask` | No | Path to binary ROI mask for loss masking / image cropping |
| `subject_id` | No | Subject identifier (required for `paired` type) |
| `modality` | No | Per-image modality (e.g., `"ct"`, `"t1"`, `"t2"`, `"flair"`). `"ct"` uses CT preprocessing, all others use MRI. Also used for label randomization grouping. |

**Consistency rule:** Every dataset entry in every dataset must provide the optional fields required by the training configuration. This ensures training stability, and the loss function composition is consistent across all batches.

If required fields are missing, config validation fails before training starts with a clear error listing the dataset and entry index.

### Example: images only
```json
{"data": [{"image": "img1.nii.gz"}, {"image": "img2.nii.gz"}]}
```

### Example: with segmentations (for Dice loss)
```json
{
  "data": [
    {"image": "img1.nii.gz", "segmentation": "seg1.nii.gz"},
    {"image": "img2.nii.gz", "segmentation": "seg2.nii.gz"}
  ]
}
```

### Example: with masks (for ROI masking)
```json
{
  "data": [
    {"image": "img1.nii.gz", "mask": "mask1.nii.gz"},
    {"image": "img2.nii.gz", "mask": "mask2.nii.gz"}
  ]
}
```

### Example: with both segmentations and masks
```json
{
  "data": [
    {"image": "img1.nii.gz", "segmentation": "seg1.nii.gz", "mask": "mask1.nii.gz"},
    {"image": "img2.nii.gz", "segmentation": "seg2.nii.gz", "mask": "mask2.nii.gz"}
  ]
}
```

## Segmentation, Masking, and Dice Loss

The forward pass accepts three types of auxiliary data, each serving a distinct purpose:

| Parameter | Data source | Purpose |
|-----------|------------|---------|
| `segmentation_A/B` | `segmentation` field in JSON | Integer label maps for Dice loss computation |
| `mask_A/B` | `mask` field in JSON | Binary ROI masks passed to similarity function |
| `label_A/B` | Auto-selected from same-subject images via `subject_id` + `modality` | Alternative similarity input for modality-invariant training (`use_label`) |

### Dice Loss

When `dice_loss_weight > 0`, the Dice loss is computed between warped and target segmentations. Only classes present in both segmentations contribute to the loss (background is excluded).

```yaml
training:
  dice_loss_weight: 0.5  # Requires 'segmentation' in JSON
```

Total loss: `L_total = lambda * L_inverse_consistency + L_similarity + dice_loss_weight * L_dice`

### Loss Function Masking

When `loss_function_masking: true`, binary masks restrict where the similarity loss is computed. This focuses registration on regions of interest.

```yaml
training:
  loss_function_masking: true  # Requires 'mask' in JSON
```

### ROI Masking (Image Cropping)

When `roi_masking: true`, images are multiplied by the binary mask after augmentation, zeroing out background regions before they enter the network.

```yaml
training:
  roi_masking: true  # Requires 'mask' in JSON
```

### Combining Features

Dice loss and masking use separate data fields, so they can be combined freely:

```yaml
training:
  dice_loss_weight: 0.5       # Uses 'segmentation' field
  loss_function_masking: true  # Uses 'mask' field
  roi_masking: true            # Uses 'mask' field
```

This requires both `segmentation` and `mask` fields in the JSON data.

### Label Randomization (`use_label`)

This reproduces the multiGradICON training strategy where the similarity loss is computed
on a randomly chosen modality image instead of the registration input image. This forces
the network to learn modality-invariant features.

```yaml
training:
  use_label: true
```

**How it works:** For each pair, the dataset picks a random modality that both subjects
have in common and uses those images as the similarity input. The network still registers
the original images, but the loss is evaluated on the randomly chosen modality.

**When to use:** This is designed for multi-sequence MRI datasets (e.g., T1, T2, FLAIR
from the same scanning session) where all sequences are co-registered. It is generally
not appropriate for CT + MRI combinations, which come from different scanners and may
not share the same coordinate space.

**JSON format:** Use `subject_id` to group co-registered images per subject. The `modality`
field identifies the sequence. Labels are always sampled from the same modality across
both subjects in a pair:

```json
{
  "data": [
    {"image": "sub01_t1.nii.gz", "subject_id": "sub01", "modality": "t1"},
    {"image": "sub01_t2.nii.gz", "subject_id": "sub01", "modality": "t2"},
    {"image": "sub01_flair.nii.gz", "subject_id": "sub01", "modality": "flair"},
    {"image": "sub02_t1.nii.gz", "subject_id": "sub02", "modality": "t1"},
    {"image": "sub02_t2.nii.gz", "subject_id": "sub02", "modality": "t2"}
  ]
}
```

For preprocessing, `"ct"` triggers CT windowing; all other modality values use MRI
quantile normalization.

If no `subject_id` is present, `use_label` is a no-op. Labels are identical to images.
For subjects with multiple scans of the same modality (e.g., longitudinal data), the label
is a randomly chosen scan from the same subject, which can still provide useful regularization.

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
    quantile_range: [0.0, 0.99]  # Normalize using quantiles
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

When a dataset contains both CT and MRI images (e.g., cross-modality registration), add a `modality` field to each entry in the JSON file. Images with `modality: "ct"` use CT windowing; all other modality values use MRI quantile normalization:

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

Both preprocessing parameters can be set in the dataset config:

```yaml
datasets:
  - name: "abdomen_mrct"
    type: "paired"
    json_file: "mrct_data.json"
    weight: 1.0
    quantile_range: [0.0, 0.99]  # Applied to MRI images
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
