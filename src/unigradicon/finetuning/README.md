# Finetuning uniGradICON on Your Data

This guide shows how to finetune uniGradICON or multiGradICON on your own data with a single YAML config file. Finetuning supports multiple datasets with weighted sampling, Dice loss with segmentations, and masks.

## Table of Contents

- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [Example with Learn2Reg Data](#example-with-learn2reg-data)
- [Preparing Your Data](#preparing-your-data)
- [Configuration](#configuration)
- [Losses and Masks](#losses-and-masks)
- [Training Options](#training-options)
- [Caching and Memory](#caching-and-memory)
- [Troubleshooting](#troubleshooting)



## Quick Start

Install the package with `pip install unigradicon`, or with `pip install -e .` from a clone of the repository. A CUDA GPU is strongly recommended, since training on a CPU is very slow.

Finetuning needs two small files. A JSON file lists your images, and a YAML config file sets the training options and points to the JSON file. A minimal pair looks like this

`my_data.json`

```json
{"data": [{"image": "/path/to/image1.nii.gz"}, {"image": "/path/to/image2.nii.gz"}]}
```

`my_config.yaml`

```yaml
experiment:
  name: "my_experiment"
  model_weights: "unigradicon"

training:
  batch_size: 2

datasets:
  - name: "my_data"
    type: "unpaired"
    json_file: "my_data.json"
```

Then start finetuning with

```bash
unigradicon-finetune --config my_config.yaml
```

The finetuned weights are saved in `results/my_experiment/checkpoints/`. The [Step-by-Step Guide](#step-by-step-guide) below explains each part, and [Example Files](#example-files) lists ready-made configs for common setups.

## Step-by-Step Guide



### 1. Prepare Your Data

Create a JSON file that lists your images, one entry per image.

```json
{
  "data": [
    {"image": "images/subject01.nii.gz"},
    {"image": "images/subject02.nii.gz"},
    {"image": "images/subject03.nii.gz"},
    {"image": "images/subject04.nii.gz"}
  ]
}
```

Paths can be absolute or relative to the JSON file, and any format ITK reads works, such as NIfTI. You need at least two images, or at least two per subject for `paired` datasets.

### 2. Create a Configuration File

Create a YAML file (e.g., `my_config.yaml`):

```yaml
experiment:
  name: "my_finetuning_experiment"
  model_weights: "unigradicon"  # Auto-downloads if not found

training:
  batch_size: 2
  gpus: [0]  # GPU device IDs
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

These values are examples. See [Training Parameters](#training-parameters) for the defaults.

### 3. Start Training

```bash
unigradicon-finetune --config my_config.yaml
```



### 4. Monitor Training

Losses and example registrations are logged to TensorBoard, with segmentation and mask overlays when your data has them. Validation always uses the same four pairs from each dataset, so its curves and images can be compared across epochs.

```bash
# Logs are written to results/<experiment.name>/logs/
tensorboard --logdir results/
```



### 5. Use Your Finetuned Model

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



### Example Files

The repository's `configs/examples/` folder has starting points for common setups. The JSON files contain placeholder paths (`/path/to/...`), so replace them with your own images before running.


| File                                                                     | Shows                                          |
| ------------------------------------------------------------------------ | ---------------------------------------------- |
| `config.yaml` + `example_unpaired.json`                                  | Images only, one MRI dataset                   |
| `config_with_seg.yaml` + `example_seg.json`                              | Dice loss with segmentations (CT)              |
| `config_multi_dataset.yaml` + `example_unpaired.json`, `example_ct.json` | An MRI and a CT dataset with weighted sampling |
| `example_paired.json`, `example_paired_seg.json`                         | `paired` datasets using `subject_id`           |
| `example_mask.json`, `example_seg_and_mask.json`                         | Masks, and masks with segmentations            |




## Example with Learn2Reg Data

This section walks through finetuning uniGradICON on three public [Learn2Reg](https://learn2reg.grand-challenge.org/) datasets. Ready-to-use configs for each dataset and for multi-dataset training are in `configs/learn2reg/`, and `scripts/prepare_l2r_datasets.py` generates the JSON files they read. Both are in the GitHub repository, not in the pip package, so run these steps from a clone of the repo.


| Config                 | Dataset            | Type       | Modality | Similarity | Pretrained    | Images              |
| ---------------------- | ------------------ | ---------- | -------- | ---------- | ------------- | ------------------- |
| `l2r_oasis.yaml`       | OASIS brain MRI    | `unpaired` | MRI      | lncc       | uniGradICON   | 414                 |
| `l2r_lungct.yaml`      | LungCT             | `paired`   | CT       | lncc       | uniGradICON   | 40 (20 x 2)         |
| `l2r_abdomenmrct.yaml` | AbdomenMRCT        | `unpaired` | CT + MR  | lncc2      | multiGradICON | 105 (57 CT + 48 MR) |
| `l2r_multi.yaml`       | All three combined | mixed      | MRI + CT | lncc2      | multiGradICON | 559                 |




### 1. Install uniGradICON

See [Quick Start](#quick-start).

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
    ├── imagesTr/       # 105 images (57 CT + 48 MR, cross-modality)
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

# Lung CT (paired, 20 subjects x 2 time points)
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
  --transform_out transform.hdf5 \
  --network_weights results/l2r_oasis_brain_mri/checkpoints/network_weights_final.trch
```



## Preparing Your Data

This part explains how image pairs are formed, what the JSON files can contain, and how CT and MRI images are preprocessed.

### Dataset Types

The dataset type controls how image pairs are formed.

- `unpaired` pairs random images from the dataset.
- `paired` pairs only images of the same subject, so every entry needs a `subject_id`.

```yaml
datasets:
  - name: "my_paired_data"
    type: "paired"
    json_file: "my_paired_data.json"
```



### JSON Data Fields

Besides the `image` path, each entry can have the optional fields below. Which ones you need depends on your config. Dice loss (`dice_loss_weight > 0`) needs a `segmentation` for every image, and `loss_function_masking` or `roi_masking` need a `mask` for every image. Segmentations and masks must be on the same voxel grid as their image, with the same size, spacing, origin, and direction. Fields your config doesn't use are ignored, and missing ones stop finetuning before training starts.


| Field          | Required | Description                                                                                                                                                                                             |
| -------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `image`        | Yes      | Path to the image file                                                                                                                                                                                  |
| `segmentation` | No       | Path to segmentation (integer labels) for Dice loss                                                                                                                                                     |
| `mask`         | No       | Path to binary ROI mask for loss masking / ROI masking                                                                                                                                                  |
| `subject_id`   | No       | Subject identifier (required for `paired` type)                                                                                                                                                         |
| `modality`     | No       | Per-image modality (e.g., `"ct"`, `"t1"`, `"t2"`, `"flair"`). Any value matching `"ct"` (case-insensitive) uses CT preprocessing, all other values use MRI. Also used for label randomization grouping. |


An entry with both optional fields looks like this. You can leave out the ones your config doesn't use.

```json
{
  "data": [
    {"image": "images/subject01.nii.gz", "segmentation": "labels/subject01.nii.gz", "mask": "masks/subject01.nii.gz"},
    {"image": "images/subject02.nii.gz", "segmentation": "labels/subject02.nii.gz", "mask": "masks/subject02.nii.gz"}
  ]
}
```



### CT and MRI Preprocessing

Before training, image intensities are scaled to [0, 1]. MRI images, the default, are scaled between two intensity quantiles, and CT images are clipped to a fixed Hounsfield unit window.

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



### Mixed CT and MRI Datasets

When a dataset contains both CT and MRI images (e.g., cross-modality registration), add a `modality` field to each entry in the JSON file. Any value matching `"ct"` (case-insensitive) uses CT windowing, and all other values use MRI quantile normalization.

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

Both preprocessing settings can be given in the dataset config.

```yaml
datasets:
  - name: "abdomen_mrct"
    type: "paired"
    json_file: "mrct_data.json"
    weight: 1.0
    quantile_range: [0.0, 0.99]  # Applied to MRI images
    ct_window: [-1000, 1000]      # Applied to CT images
```

If `modality` is not specified for an entry, the dataset-level `is_ct` setting is used as the fallback.

### Matching Preprocessing at Inference

The default preprocessing is the same for finetuning and inference. If you changed `quantile_range` or `ct_window` in your config, pass the same values to `unigradicon-register`. The CLI only accepts `ct` and `mri` as modalities, so use `mri` for any non-CT image such as T1 or FLAIR.

```bash
# Example: custom ct_window used during finetuning
unigradicon-register \
  --fixed fixed.nii.gz --moving moving.nii.gz \
  --fixed_modality ct --moving_modality ct \
  --ct_window -500 1500 \
  --transform_out transform.hdf5 \
  --network_weights results/my_experiment/checkpoints/network_weights_final.trch
```



## Configuration

The tables below list every setting in the YAML config with its default, followed by notes on augmentation and the input shape.

### Training Parameters


| Parameter                  | Type  | Description                                                                                      | Default         |
| -------------------------- | ----- | ------------------------------------------------------------------------------------------------ | --------------- |
| `batch_size`               | int   | Registration pairs per GPU                                                                       | 4               |
| `gpus`                     | list  | GPU device IDs                                                                                   | [0]             |
| `epochs`                   | int   | Training epochs                                                                                  | 500             |
| `learning_rate`            | float | Adam learning rate                                                                               | 5e-5            |
| `input_shape`              | list  | Model input dimensions [D, H, W] (images are resampled to this)                                  | [175, 175, 175] |
| `eval_period`              | int   | Validate every N epochs                                                                          | 10              |
| `save_period`              | int   | Save checkpoint every N epochs                                                                   | 50              |
| `seed`                     | int   | Random seed for reproducibility                                                                  | null            |
| `random_flips`             | bool  | Apply random axis flips during augmentation                                                      | true            |
| `random_axis_permutations` | bool  | Randomly swap spatial axes during augmentation                                                   | true            |
| `lambda`                   | float | Regularization weight                                                                            | 1.5             |
| `similarity`               | str   | Loss function: "lncc", "lncc2", "mind"                                                           | "lncc"          |
| `dice_loss_weight`         | float | Dice loss weight (requires `segmentation` in JSON)                                               | 0.0             |
| `loss_function_masking`    | bool  | Restrict similarity loss to masked regions (requires `mask` in JSON)                             | false           |
| `roi_masking`              | bool  | Zero out image background outside the ROI before registration (requires `mask` in JSON)          | false           |
| `use_label`                | bool  | Label randomization for modality-invariant training (see below)                                  | false           |
| `lncc_sigma`               | int   | Sigma for LNCC / SquaredLNCC similarity                                                          | 5               |
| `mind_radius`              | int   | Radius for MIND-SSC similarity                                                                   | 2               |
| `mind_dilation`            | int   | Dilation for MIND-SSC similarity                                                                 | 2               |
| `samples_per_epoch`        | int   | Number of samples drawn per epoch (with replacement); null defaults to the combined dataset size | null            |
| `num_workers`              | int   | DataLoader worker processes                                                                      | 4               |




### Dataset Parameters


| Parameter         | Type  | Description                                                                                         | Default       |
| ----------------- | ----- | --------------------------------------------------------------------------------------------------- | ------------- |
| `name`            | str   | Dataset identifier                                                                                  | Required      |
| `type`            | str   | `"unpaired"` or `"paired"`                                                                          | Required      |
| `json_file`       | str   | Path to JSON dataset definition                                                                     | Required      |
| `weight`          | float | Relative sampling weight                                                                            | 1.0           |
| `maximum_images`  | int   | Limit number of images                                                                              | null          |
| `use_cache`       | bool  | Enable/disable caching                                                                              | true          |
| `use_compression` | bool  | Compress images in RAM and on disk with blosc (see [In-Memory Compression](#in-memory-compression)) | false         |
| `cache_dir`       | str   | Directory for cached datasets                                                                       | null          |
| `shuffle`         | bool  | Shuffle image order before loading                                                                  | true          |
| `is_ct`           | bool  | CT or MRI preprocessing                                                                             | false         |
| `ct_window`       | list  | HU window for CT [min, max]                                                                         | [-1000, 1000] |
| `quantile_range`  | list  | Intensity quantile range for MRI                                                                    | [0.0, 0.99]   |


`json_file` paths are resolved relative to the YAML file, so you can usually give just the filename.

### Augmentation

Every training pair gets a small random affine change. By default, the image axes are also randomly flipped and swapped. If all your images have the same orientation, you can turn this off, as the example configs do

```yaml
training:
  random_flips: false
  random_axis_permutations: false
```



### Input Shape

The released weights were trained at `input_shape: [175, 175, 175]`, and `unigradicon-register` always runs at that size. Keep this shape if you plan to use the finetuned weights with the CLI. Other shapes might need more epochs to converge.

## Losses and Masks

Segmentations add a Dice loss, and masks can restrict the similarity loss or remove the image background outside the region of interest. Both are optional and come from the `segmentation` and `mask` fields in your JSON files.

### Dice Loss

When `dice_loss_weight > 0`, the Dice loss is computed between warped and target segmentations. Only classes present in both segmentations contribute to the loss (background is excluded). Every sampled pair must share at least one non-background label, otherwise training stops with an error. With `unpaired` datasets any two images can be paired, so make sure all segmentations share at least one label.

```yaml
training:
  dice_loss_weight: 0.5  # Requires 'segmentation' in JSON
```

The total loss is `L_total = lambda * L_inverse_consistency + L_similarity + dice_loss_weight * L_dice`.

### Loss Function Masking

When `loss_function_masking: true`, binary masks restrict where the similarity loss is computed. This focuses registration on regions of interest.

```yaml
training:
  loss_function_masking: true  # Requires 'mask' in JSON
```



### ROI Masking

When `roi_masking: true`, images are multiplied by the binary mask after augmentation, zeroing out background regions before they enter the network.

```yaml
training:
  roi_masking: true  # Requires 'mask' in JSON
```



### Combining Features

Dice loss and masking use separate JSON fields, so you can combine them.

```yaml
training:
  dice_loss_weight: 0.5       # Uses 'segmentation' field
  loss_function_masking: true  # Uses 'mask' field
  roi_masking: true            # Uses 'mask' field
```



### Label Randomization (`use_label`)

This reproduces the multiGradICON training strategy. The network registers the input images, but the similarity loss is computed on another image of the same two subjects, in a modality both of them have.

```yaml
training:
  use_label: true
```

It is meant for co-registered multi-sequence MRI, for example T1, T2, and FLAIR from one session. It is usually not suitable for CT with MRI, which rarely share a coordinate space. Group the images of each subject with `subject_id` and name the sequence with `modality`.

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

Without `subject_id`, `use_label` has no effect.

## Training Options



### Starting Weights

`model_weights` sets the weights to start from. Use `unigradicon` or `multigradicon` to download the pretrained weights, or give a path to your own.

```yaml
experiment:
  model_weights: "unigradicon"      # Auto-downloads uniGradICON weights
  # OR
  model_weights: "multigradicon"    # Auto-downloads multiGradICON weights
  # OR
  model_weights: "/path/to/my/weights.trch"  # Use custom weights
```



### Warm Start From a Checkpoint

To continue from a saved checkpoint, point `model_weights` at it.

```yaml
experiment:
  model_weights: "results/my_experiment/checkpoints/network_weights_50.trch"
```

If the matching `optimizer_weights_50.trch` is in the same folder, the optimizer state is loaded too, but the `learning_rate` from your config is used. Epochs count from 0 again and logs go to a new run, so set `epochs` to the number of additional epochs you want.

### Multi-Dataset Training

To train on several datasets at once, list them with sampling weights.

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

Weights are relative, so they don't need to sum to 1.

### Samples Per Epoch

By default, each epoch draws as many samples as the datasets contain. Set `samples_per_epoch` to change this, for example for quick tests. Samples are drawn with replacement.

```yaml
training:
  samples_per_epoch: 4000
```



### Reproducibility

Set a seed to make runs repeatable.

```yaml
training:
  seed: 42
```

Results can still differ slightly between runs, because data loading with several workers and some GPU operations are not deterministic.

## Caching and Memory



### Caching

Preprocessed images are cached so later runs start faster. The cache is stored in `cache_dir`, or in the experiment's results folder if `cache_dir` is not set, and it is rebuilt automatically when the dataset, its files, or a preprocessing setting changes. To turn it off, for example while your data is still changing

```yaml
datasets:
  - name: "my_dataset"
    use_cache: false
```

When running several jobs in parallel, we recommend giving each job its own `cache_dir`, or running one job first to build the cache.

### In-Memory Compression

If the preprocessed dataset doesn't fit in RAM, set `use_compression: true`. Images, segmentations, and masks are then stored compressed with [blosc](https://www.blosc.org/) and decompressed per sample.

```yaml
datasets:
  - name: "large_3d_dataset"
    use_compression: true
```



## Troubleshooting

Here are some common issues that may arise during finetuning, along with their solutions.

- If you see "JSON file not found", check the json_file path. Relative paths are resolved from the folder of the YAML file, so use an absolute path if the JSON file is somewhere else.
- The JSON file must have the form `{"data": [...]}`. Otherwise finetuning stops with "must contain top-level 'data' key".
- "has no data entries" means the data list in the JSON file is empty. Add your images to it.
- If a setting in your YAML file is misspelled, a warning lists it as unrecognized and suggests the closest valid key, for example "'epocs' (did you mean 'epochs'?)". The misspelled setting has no effect until you fix it.
- Errors such as "must be a number" or "must be true or false" mean a value has the wrong type. YAML reads 5e-5 as text, so write `5.0e-5` instead, and write booleans without quotes (`false`, not `"false"`).
- Dataset weights do not need to sum to 1, since they are relative. They only need to be positive.
- If the cache uses too much disk space, delete old cache folders in the cache_dir (or in the experiment's results folder), set `use_cache: false`, or limit maximum_images.
