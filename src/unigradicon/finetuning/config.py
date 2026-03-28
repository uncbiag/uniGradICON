import copy
import logging
import yaml
import json
import os
from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader
from typing import Dict, List, Tuple, Any
from . import dataset

logger = logging.getLogger(__name__)

REQUIRED_EXPERIMENT_KEYS = {'name', 'model_weights'}
REQUIRED_DATASET_KEYS = {'name', 'type', 'json_file'}
VALID_TRAINING_KEYS = {
    'batch_size', 'gpus', 'epochs', 'eval_period', 'save_period', 'learning_rate',
    'input_shape', 'seed', 'similarity', 'lambda', 'dice_loss_weight',
    'loss_function_masking', 'lncc_sigma', 'mind_radius', 'mind_dilation',
    'samples_per_epoch', 'num_workers',
}
VALID_DATASET_KEYS = {
    'name', 'type', 'json_file', 'weight', 'maximum_images', 'use_cache',
    'cache_dir', 'is_ct', 'ct_window', 'quantile_range', 'read_type', 'shuffle',
}


def validate_config(config: Dict[str, Any]):
    """Validate config schema and warn about unrecognized keys."""
    if 'experiment' not in config:
        raise ValueError("Config must contain 'experiment' section")
    if 'datasets' not in config or not config['datasets']:
        raise ValueError("Config must contain non-empty 'datasets' section")

    exp = config['experiment']
    for key in REQUIRED_EXPERIMENT_KEYS:
        if key not in exp:
            raise ValueError(f"Missing required experiment key: '{key}'")

    if 'training' in config:
        unknown = set(config['training'].keys()) - VALID_TRAINING_KEYS
        if unknown:
            logger.warning(f"Unrecognized training keys (possible typo): {unknown}")

    for i, ds in enumerate(config['datasets']):
        for key in REQUIRED_DATASET_KEYS:
            if key not in ds:
                raise ValueError(f"Dataset {i} ('{ds.get('name', '?')}'): missing required key '{key}'")
        unknown = set(ds.keys()) - VALID_DATASET_KEYS
        if unknown:
            logger.warning(f"Dataset '{ds.get('name', '?')}': unrecognized keys (possible typo): {unknown}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_json_dataset_file(json_path: str) -> List[Dict[str, str]]:
    """Load dataset definition from JSON file. Returns a new list (does not mutate input)."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON dataset file not found: {json_path}")

    with open(json_path, 'r') as f:
        content = json.load(f)

    if "data" not in content:
        raise ValueError(f"JSON dataset file {json_path} must contain top-level 'data' key.")

    data = [dict(item) for item in content["data"]]
    base_dir = os.path.dirname(os.path.abspath(json_path))

    for idx, item in enumerate(data):
        if "image" not in item:
            raise ValueError(f"Missing 'image' field in entry {idx} of {json_path}")

        image_path = item["image"]
        resolved_image = image_path if os.path.isabs(image_path) else os.path.join(base_dir, image_path)
        if not os.path.exists(resolved_image):
            raise FileNotFoundError(
                f"Image file not found for entry {idx} in {json_path}: "
                f"{image_path} (resolved to {resolved_image})"
            )
        item["image"] = resolved_image

        if "segmentation" in item:
            seg_path = item["segmentation"]
            resolved_seg = seg_path if os.path.isabs(seg_path) else os.path.join(base_dir, seg_path)
            if not os.path.exists(resolved_seg):
                raise FileNotFoundError(
                    f"Segmentation file not found for entry {idx} in {json_path}: "
                    f"{seg_path} (resolved to {resolved_seg})"
                )
            item["segmentation"] = resolved_seg

    return data


def validate_dataset_consistency(configs: List[Dict[str, Any]]) -> str:
    """
    Validate that all datasets are consistent (either all with segmentation or all without).
    Returns: 'image' for datasets without segmentation, 'segmentation' for datasets with segmentation
    Raises: ValueError if datasets are mixed
    """
    seg_types = {'unpaired_with_seg', 'paired_with_seg'}
    image_types = {'unpaired', 'paired'}

    dataset_types = [ds['type'] for ds in configs]

    has_seg = any(dt in seg_types for dt in dataset_types)
    has_image = any(dt in image_types for dt in dataset_types)

    if has_seg and has_image:
        raise ValueError(
            "Cannot mix dataset types with and without segmentations in the same config. "
            f"Found types: {dataset_types}. "
            "Use either all image types (unpaired, paired) or all segmentation types "
            "(unpaired_with_seg, paired_with_seg)."
        )

    if has_seg:
        return 'segmentation'
    else:
        return 'image'


def create_dataset_from_config(dataset_config: Dict[str, Any], input_shape: Tuple[int, ...], config_dir: str = "") -> dataset.Dataset:
    """
    Instantiate a dataset based on config using existing classes:
    - Dataset (unpaired)
    - PairedDataset
    - ImageSegmentationDataset (unpaired_with_seg)
    - PairedImageSegmentationDataset (paired_with_seg)
    """
    dataset_type = dataset_config['type']

    common_params = {
        'input_shape': input_shape,
        'name': dataset_config['name'],
        'read_type': dataset_config.get('read_type', 'itk'),
        'cache_dir': dataset_config.get('cache_dir'),
        'maximum_images': dataset_config.get('maximum_images'),
        'shuffle': dataset_config.get('shuffle', True),
        'is_ct': dataset_config.get('is_ct', False),
        'use_cache': dataset_config.get('use_cache', True),
    }

    common_params['ct_window'] = tuple(dataset_config.get('ct_window', [-1000, 1000]))
    common_params['quantile_range'] = tuple(dataset_config.get('quantile_range', [0.01, 0.99]))

    json_file = dataset_config['json_file']
    if config_dir and not os.path.isabs(json_file):
        json_file = os.path.join(config_dir, json_file)
    common_params['data'] = load_json_dataset_file(json_file)

    if dataset_type == 'unpaired':
        return dataset.Dataset(**common_params)
    elif dataset_type == 'paired':
        return dataset.PairedDataset(**common_params)
    elif dataset_type == 'unpaired_with_seg':
        return dataset.ImageSegmentationDataset(**common_params)
    elif dataset_type == 'paired_with_seg':
        return dataset.PairedImageSegmentationDataset(**common_params)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Must be one of: unpaired, paired, unpaired_with_seg, paired_with_seg")


def create_data_loaders(config_path: str, config: Dict[str, Any] = None) -> Tuple[DataLoader, Dict[str, DataLoader], Dict[str, Any], str]:
    """
    Create training and validation dataloaders from YAML config.

    Args:
        config_path: Path to YAML config file (used to resolve relative paths)
        config: Pre-loaded config dict. If None, loads from config_path.

    Returns:
        train_loader: DataLoader for training with weighted sampling
        val_loaders: Dict mapping dataset name to its validation DataLoader
        config: The loaded configuration dictionary
        mode: 'image' or 'segmentation' indicating dataset type
    """
    if config is None:
        config = load_config(config_path)
        validate_config(config)
    config = copy.deepcopy(config)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    training_defaults = {
        'batch_size': 4,
        'gpus': [0],
        'epochs': 500,
        'eval_period': 15,
        'save_period': 50,
        'input_shape': [175, 175, 175],
    }
    train_config = config.setdefault('training', {})
    for k, v in training_defaults.items():
        train_config.setdefault(k, v)

    mode = validate_dataset_consistency(config['datasets'])
    logger.info(f"Dataset mode: {mode}")

    input_shape = train_config['input_shape']
    batch_size = train_config['batch_size']
    gpus = train_config['gpus']
    num_gpus = len(gpus)
    num_workers = train_config.get('num_workers', 4)

    datasets = []
    weights = []
    val_loaders = {}

    logger.info(f"Loading {len(config['datasets'])} dataset(s)...")
    for ds_config in config['datasets']:
        logger.info(f"Processing dataset: {ds_config['name']} (type={ds_config['type']}, weight={ds_config.get('weight', 1.0)})")

        ds = create_dataset_from_config(ds_config, input_shape, config_dir=config_dir)
        ds.compress()
        datasets.append(ds)

        if hasattr(ds, '__len__'):
            ds_length = len(ds)
        elif hasattr(ds, 'keys'):
            ds_length = len(ds.keys) if ds.keys else 0
        else:
            raise ValueError(f"Dataset {ds_config['name']} has no way to determine length (no __len__ or keys attribute)")

        if ds_length == 0:
            raise ValueError(f"Dataset {ds_config['name']} is empty; cannot build sampler.")

        ds_weight = ds_config.get('weight', 1.0)
        per_sample_weight = ds_weight / ds_length
        weights.extend([per_sample_weight] * ds_length)

        logger.info(f"Loaded {ds_length} samples")

        val_loaders[ds_config['name']] = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )

    combined_dataset = ConcatDataset(datasets)
    total_samples = len(combined_dataset)

    samples_per_epoch = config['training'].get('samples_per_epoch', total_samples)

    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size * num_gpus,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        sampler=WeightedRandomSampler(
            weights=normalized_weights,
            num_samples=samples_per_epoch,
            replacement=True
        )
    )

    effective_batch_size = batch_size * num_gpus
    iterations_per_epoch = samples_per_epoch // effective_batch_size

    logger.info(f"Total samples: {total_samples} | Samples/epoch: {samples_per_epoch} | "
                f"Batch: {batch_size}x{num_gpus} GPUs | Iters/epoch: {iterations_per_epoch}")

    return train_loader, val_loaders, config, mode
