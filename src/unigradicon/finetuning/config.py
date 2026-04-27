import copy
import logging
import yaml
import json
import os
from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader
from typing import Dict, List, Tuple, Any, FrozenSet
from . import dataset

logger = logging.getLogger(__name__)

REQUIRED_EXPERIMENT_KEYS = {'name', 'model_weights'}
REQUIRED_DATASET_KEYS = {'name', 'type', 'json_file'}
OPTIONAL_DATA_FIELDS = frozenset({"segmentation", "mask"})
VALID_TRAINING_KEYS = {
    'batch_size', 'gpus', 'epochs', 'eval_period', 'save_period', 'learning_rate',
    'input_shape', 'seed', 'similarity', 'lambda', 'dice_loss_weight',
    'loss_function_masking', 'roi_masking', 'use_label', 'lncc_sigma', 'mind_radius',
    'mind_dilation', 'samples_per_epoch', 'num_workers',
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

        for field in ("segmentation", "mask"):
            if field in item:
                path = item[field]
                resolved = path if os.path.isabs(path) else os.path.join(base_dir, path)
                if not os.path.exists(resolved):
                    raise FileNotFoundError(
                        f"{field.capitalize()} file not found for entry {idx} in {json_path}: "
                        f"{path} (resolved to {resolved})"
                    )
                item[field] = resolved

    return data


def required_data_fields(train_config: Dict[str, Any]) -> FrozenSet[str]:
    """Determine which optional data fields are required by the training config."""
    fields = set()
    if train_config.get('dice_loss_weight', 0.0) > 0.0:
        fields.add("segmentation")
    if train_config.get('loss_function_masking', False) or train_config.get('roi_masking', False):
        fields.add("mask")
    return frozenset(fields)


def determine_data_fields(dataset_configs: List[Dict[str, Any]], config_dir: str) -> Dict[str, List[FrozenSet[str]]]:
    """Determine which optional data fields (segmentation, mask) are present.

    Returns a mapping from dataset name to per-entry field sets. Validation of
    whether these fields are sufficient is driven by the training config.
    """
    fields_per_dataset = {}

    for ds_config in dataset_configs:
        json_file = ds_config['json_file']
        if config_dir and not os.path.isabs(json_file):
            json_file = os.path.join(config_dir, json_file)

        with open(json_file, 'r') as f:
            content = json.load(f)

        if "data" not in content or not content["data"]:
            raise ValueError(f"JSON file {json_file} has no data entries")

        fields_per_dataset[ds_config['name']] = [
            frozenset(k for k in OPTIONAL_DATA_FIELDS if k in entry)
            for entry in content["data"]
        ]

    return fields_per_dataset


def validate_training_data_compatibility(
    fields_per_dataset: Dict[str, List[FrozenSet[str]]],
    data_fields: FrozenSet[str],
):
    """Cross-validate training config requirements against available data fields."""
    for dataset_name, entry_fields in fields_per_dataset.items():
        for idx, fields in enumerate(entry_fields):
            missing = data_fields - fields
            if missing:
                raise ValueError(
                    f"Dataset '{dataset_name}' entry {idx} is missing required field(s) "
                    f"{sorted(missing)} for the current training config."
                )

    available_fields = frozenset().union(
        *(fields for entry_fields in fields_per_dataset.values() for fields in entry_fields)
    ) if fields_per_dataset else frozenset()
    ignored_fields = available_fields - data_fields
    if ignored_fields:
        logger.warning(
            f"Ignoring optional JSON field(s) not required by training config: {sorted(ignored_fields)}"
        )


def _keep_required_optional_fields(data: List[Dict[str, str]], data_fields: FrozenSet[str]) -> List[Dict[str, str]]:
    """Drop optional fields that are not required by this training run."""
    keep_fields = OPTIONAL_DATA_FIELDS & data_fields
    filtered = []
    for item in data:
        filtered_item = dict(item)
        for field in OPTIONAL_DATA_FIELDS - keep_fields:
            filtered_item.pop(field, None)
        filtered.append(filtered_item)
    return filtered


def create_dataset_from_config(dataset_config: Dict[str, Any], input_shape: Tuple[int, ...],
                               config_dir: str = "", use_label: bool = False,
                               data_fields: FrozenSet[str] = frozenset()) -> dataset.Dataset:
    """Instantiate a dataset based on config.

    Dataset type determines pairing strategy:
    - ``unpaired``: random pairing from all images
    - ``paired``: subject-based pairing (requires ``subject_id`` in JSON)

    What auxiliary data is loaded (segmentations, masks) is determined by
    the training config's required data fields, not by the dataset type.
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
        'use_label': use_label,
    }

    common_params['ct_window'] = tuple(dataset_config.get('ct_window', [-1000, 1000]))
    common_params['quantile_range'] = tuple(dataset_config.get('quantile_range', [0.0, 0.99]))

    json_file = dataset_config['json_file']
    if config_dir and not os.path.isabs(json_file):
        json_file = os.path.join(config_dir, json_file)
    common_params['data'] = _keep_required_optional_fields(
        load_json_dataset_file(json_file), data_fields
    )

    if dataset_type == 'unpaired':
        return dataset.Dataset(**common_params)
    elif dataset_type == 'paired':
        return dataset.PairedDataset(**common_params)
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Must be 'unpaired' or 'paired'. Data fields (segmentation, mask) "
            f"are auto-detected from JSON entries."
        )


def create_data_loaders(config_path: str, config: Dict[str, Any] = None) -> Tuple[DataLoader, Dict[str, DataLoader], Dict[str, Any], FrozenSet[str]]:
    """
    Create training and validation dataloaders from YAML config.

    Args:
        config_path: Path to YAML config file (used to resolve relative paths)
        config: Pre-loaded config dict. If None, loads from config_path.

    Returns:
        train_loader: DataLoader for training with weighted sampling
        val_loaders: Dict mapping dataset name to its validation DataLoader
        config: The loaded configuration dictionary
        data_fields: Frozenset of optional data fields required by the config
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
        'eval_period': 10,
        'save_period': 50,
        'input_shape': [175, 175, 175],
    }
    train_config = config.setdefault('training', {})
    for k, v in training_defaults.items():
        train_config.setdefault(k, v)

    data_fields = required_data_fields(train_config)
    fields_per_dataset = determine_data_fields(config['datasets'], config_dir)
    validate_training_data_compatibility(fields_per_dataset, data_fields)
    logger.info(f"Required data fields: {data_fields or 'images only'}")

    input_shape = train_config['input_shape']
    batch_size = train_config['batch_size']
    gpus = train_config['gpus']
    num_gpus = len(gpus)
    num_workers = train_config.get('num_workers', 4)
    use_label = train_config.get('use_label', False)

    if use_label:
        has_subject_ids = False
        for ds_config in config['datasets']:
            json_file = ds_config['json_file']
            if config_dir and not os.path.isabs(json_file):
                json_file = os.path.join(config_dir, json_file)
            with open(json_file, 'r') as f:
                entries = json.load(f).get("data", [])
            if any('subject_id' in e for e in entries):
                has_subject_ids = True
                break
        if not has_subject_ids:
            logger.warning(
                "use_label is enabled but no 'subject_id' found in any dataset. "
                "Without subject_id, labels will be identical to images (no-op)."
            )

    datasets = []
    weights = []
    val_loaders = {}

    logger.info(f"Loading {len(config['datasets'])} dataset(s)...")
    for ds_config in config['datasets']:
        logger.info(f"Processing dataset: {ds_config['name']} (type={ds_config['type']}, weight={ds_config.get('weight', 1.0)})")

        ds = create_dataset_from_config(
            ds_config,
            input_shape,
            config_dir=config_dir,
            use_label=use_label,
            data_fields=data_fields,
        )
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

    return train_loader, val_loaders, config, data_fields
