import copy
import logging
import random
import yaml
import json
import os
from dataclasses import dataclass, field, fields
import numpy as np
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader
from typing import Dict, List, Tuple, Any, FrozenSet, Optional
from . import dataset

logger = logging.getLogger(__name__)


def _seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: seeds Python ``random`` and NumPy from
    PyTorch's per-worker seed so augmentation/sampling done inside worker
    subprocesses is reproducible when the parent seed is set."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def set_reproducibility_seed(seed: Optional[int]) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and GPU) at startup."""
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


class ConfigSections:
    EXPERIMENT = 'experiment'
    TRAINING = 'training'
    DATASETS = 'datasets'


class ExperimentKeys:
    NAME = 'name'
    MODEL_WEIGHTS = 'model_weights'


class TrainingKeys:
    BATCH_SIZE = 'batch_size'
    GPUS = 'gpus'
    EPOCHS = 'epochs'
    EVAL_PERIOD = 'eval_period'
    SAVE_PERIOD = 'save_period'
    LEARNING_RATE = 'learning_rate'
    INPUT_SHAPE = 'input_shape'
    SEED = 'seed'
    SIMILARITY = 'similarity'
    LAMBDA = 'lambda'
    DICE_LOSS_WEIGHT = 'dice_loss_weight'
    LOSS_FUNCTION_MASKING = 'loss_function_masking'
    ROI_MASKING = 'roi_masking'
    USE_LABEL = 'use_label'
    LNCC_SIGMA = 'lncc_sigma'
    MIND_RADIUS = 'mind_radius'
    MIND_DILATION = 'mind_dilation'
    SAMPLES_PER_EPOCH = 'samples_per_epoch'
    NUM_WORKERS = 'num_workers'


class DatasetKeys:
    NAME = 'name'
    TYPE = 'type'
    JSON_FILE = 'json_file'
    WEIGHT = 'weight'
    MAXIMUM_IMAGES = 'maximum_images'
    USE_CACHE = 'use_cache'
    USE_COMPRESSION = 'use_compression'
    CACHE_DIR = 'cache_dir'
    IS_CT = 'is_ct'
    CT_WINDOW = 'ct_window'
    QUANTILE_RANGE = 'quantile_range'
    SHUFFLE = 'shuffle'


class DatasetTypes:
    UNPAIRED = 'unpaired'
    PAIRED = 'paired'


class JsonKeys:
    DATA = 'data'


REQUIRED_EXPERIMENT_KEYS = {ExperimentKeys.NAME, ExperimentKeys.MODEL_WEIGHTS}
REQUIRED_DATASET_KEYS = {DatasetKeys.NAME, DatasetKeys.TYPE, DatasetKeys.JSON_FILE}
OPTIONAL_DATA_FIELDS = frozenset({dataset.Fields.SEGMENTATION, dataset.Fields.MASK})

# Must mirror unigradicon.make_sim's accepted values; runtime lowercases before
# dispatching, so this set is the canonical lowercase form.
VALID_SIMILARITIES = frozenset({"lncc", "lncc2", "mind"})

# YAML uses "lambda" but ``lambda`` is a Python keyword, so the dataclass
# field is named ``lmbda``; this alias bridges the two.
TRAINING_FIELD_ALIASES = {TrainingKeys.LAMBDA: "lmbda"}
DATASET_FIELD_ALIASES: Dict[str, str] = {}

DEFAULT_VAL_BATCH_SIZE = 1
DEFAULT_DROP_LAST = True
DEFAULT_PIN_MEMORY = True
DEFAULT_SAMPLER_REPLACEMENT = True


def _schema_kwargs(raw: Dict[str, Any], schema_cls, aliases: Dict[str, str]) -> Dict[str, Any]:
    """Translate a raw YAML dict into kwargs for a dataclass.

    Forwards only keys that map to a dataclass field (directly or via alias);
    missing keys fall through to the dataclass field defaults. Unknown keys
    are silently dropped here — ``ConfigValidator`` already warns about them.
    """
    valid_field_names = {f.name for f in fields(schema_cls)}
    kwargs: Dict[str, Any] = {}
    for key, value in raw.items():
        field_name = aliases.get(key, key)
        if field_name in valid_field_names:
            kwargs[field_name] = value
    return kwargs


def _yaml_keys_for_dataclass(schema_cls, aliases: Dict[str, str]) -> set:
    inverse = {field_name: yaml_key for yaml_key, field_name in aliases.items()}
    return {inverse.get(f.name, f.name) for f in fields(schema_cls)}


@dataclass
class ExperimentConfig:
    name: str
    model_weights: str

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            name=raw[ExperimentKeys.NAME],
            model_weights=raw[ExperimentKeys.MODEL_WEIGHTS],
        )


@dataclass
class TrainingConfig:
    batch_size: int = 4
    gpus: List[int] = field(default_factory=lambda: [0])
    epochs: int = 500
    eval_period: int = 10
    save_period: int = 50
    input_shape: List[int] = field(default_factory=lambda: [175, 175, 175])
    learning_rate: float = 0.00005
    num_workers: int = 4
    use_label: bool = False
    samples_per_epoch: Optional[int] = None
    lmbda: float = 1.5
    similarity: str = "lncc"
    lncc_sigma: int = 5
    mind_radius: int = 2
    mind_dilation: int = 2
    dice_loss_weight: float = 0.0
    loss_function_masking: bool = False
    roi_masking: bool = False
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TrainingConfig":
        kwargs = _schema_kwargs(raw, cls, TRAINING_FIELD_ALIASES)
        if "gpus" in kwargs:
            kwargs["gpus"] = list(kwargs["gpus"])
        if "input_shape" in kwargs:
            kwargs["input_shape"] = list(kwargs["input_shape"])
        return cls(**kwargs)

    @property
    def network_input_shape(self) -> List[int]:
        """``input_shape`` with the [1, 1] batch+channel prefix that
        ``unigradicon.make_network`` expects."""
        return [1, 1] + list(self.input_shape)


@dataclass
class DatasetConfig(dataset.DatasetParams):
    """``name``/``type``/``json_file`` are YAML-required and validated in
    ``__post_init__``; the empty defaults exist only because dataclass
    inheritance forbids non-defaulted fields after the all-defaulted
    ``DatasetParams`` parent."""
    name: str = ""
    type: str = ""
    json_file: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        for field_name in ("name", "type", "json_file"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"DatasetConfig requires non-empty '{field_name}' "
                    f"(got an empty value). Build via DatasetConfig.from_dict "
                    f"after validate_config()."
                )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DatasetConfig":
        kwargs = _schema_kwargs(raw, cls, DATASET_FIELD_ALIASES)
        if "ct_window" in kwargs:
            kwargs["ct_window"] = tuple(kwargs["ct_window"])
        if "quantile_range" in kwargs:
            kwargs["quantile_range"] = tuple(kwargs["quantile_range"])
        return cls(**kwargs)


@dataclass
class FinetuningConfigSchema:
    experiment: ExperimentConfig
    training: TrainingConfig
    datasets: List[DatasetConfig]

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "FinetuningConfigSchema":
        return cls(
            experiment=ExperimentConfig.from_dict(raw[ConfigSections.EXPERIMENT]),
            training=TrainingConfig.from_dict(raw.get(ConfigSections.TRAINING, {})),
            datasets=[
                DatasetConfig.from_dict(ds_config)
                for ds_config in raw[ConfigSections.DATASETS]
            ],
        )


VALID_TRAINING_KEYS = _yaml_keys_for_dataclass(TrainingConfig, TRAINING_FIELD_ALIASES)
VALID_DATASET_KEYS = _yaml_keys_for_dataclass(DatasetConfig, DATASET_FIELD_ALIASES)


@dataclass
class DataLoaderBundle:
    train_loader: DataLoader
    val_loaders: Dict[str, DataLoader]
    config: Dict[str, Any]
    data_fields: FrozenSet[str]


class ConfigValidator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def validate(self) -> None:
        self._validate_required_sections()
        self._validate_experiment()
        self._validate_training()
        self._validate_datasets()

    def _validate_required_sections(self) -> None:
        if ConfigSections.EXPERIMENT not in self.config:
            raise ValueError("Config must contain 'experiment' section")
        if ConfigSections.DATASETS not in self.config or not self.config[ConfigSections.DATASETS]:
            raise ValueError("Config must contain non-empty 'datasets' section")

    def _validate_experiment(self) -> None:
        exp = self.config[ConfigSections.EXPERIMENT]
        for key in REQUIRED_EXPERIMENT_KEYS:
            if key not in exp:
                raise ValueError(f"Missing required experiment key: '{key}'")

    def _validate_training(self) -> None:
        if ConfigSections.TRAINING not in self.config:
            return
        train_config = self.config[ConfigSections.TRAINING]
        unknown = set(train_config.keys()) - VALID_TRAINING_KEYS
        if unknown:
            logger.warning(f"Unrecognized training keys (possible typos): {sorted(unknown)}.")
        if TrainingKeys.LEARNING_RATE in train_config and train_config[TrainingKeys.LEARNING_RATE] <= 0:
            raise ValueError(f"'{TrainingKeys.LEARNING_RATE}' must be positive")
        for positive_key in (TrainingKeys.EVAL_PERIOD, TrainingKeys.SAVE_PERIOD,
                             TrainingKeys.EPOCHS, TrainingKeys.BATCH_SIZE,
                             TrainingKeys.LNCC_SIGMA, TrainingKeys.MIND_RADIUS,
                             TrainingKeys.MIND_DILATION):
            if positive_key in train_config and train_config[positive_key] <= 0:
                raise ValueError(
                    f"'{positive_key}' must be a positive integer "
                    f"(got {train_config[positive_key]})"
                )
        for non_negative_key in (TrainingKeys.LAMBDA, TrainingKeys.DICE_LOSS_WEIGHT,
                                 TrainingKeys.NUM_WORKERS):
            if non_negative_key in train_config and train_config[non_negative_key] < 0:
                raise ValueError(
                    f"'{non_negative_key}' must be non-negative "
                    f"(got {train_config[non_negative_key]})"
                )
        if TrainingKeys.SAMPLES_PER_EPOCH in train_config:
            spe = train_config[TrainingKeys.SAMPLES_PER_EPOCH]
            if spe is not None and (not isinstance(spe, int) or spe <= 0):
                raise ValueError(
                    f"'{TrainingKeys.SAMPLES_PER_EPOCH}' must be a positive "
                    f"integer or null (got {spe})"
                )
        if TrainingKeys.INPUT_SHAPE in train_config:
            shape = train_config[TrainingKeys.INPUT_SHAPE]
            if not (isinstance(shape, (list, tuple)) and len(shape) == 3
                    and all(isinstance(d, int) and d > 0 for d in shape)):
                raise ValueError(
                    f"'{TrainingKeys.INPUT_SHAPE}' must be a length-3 list of "
                    f"positive integers (got {shape})"
                )
        if TrainingKeys.GPUS in train_config:
            gpus = train_config[TrainingKeys.GPUS]
            if not (isinstance(gpus, (list, tuple)) and len(gpus) > 0
                    and all(isinstance(g, int) and g >= 0 for g in gpus)):
                raise ValueError(
                    f"'{TrainingKeys.GPUS}' must be a non-empty list of "
                    f"non-negative integers (got {gpus})"
                )
        if TrainingKeys.SIMILARITY in train_config:
            sim = str(train_config[TrainingKeys.SIMILARITY]).lower()
            if sim not in VALID_SIMILARITIES:
                raise ValueError(
                    f"'{TrainingKeys.SIMILARITY}' must be one of "
                    f"{sorted(VALID_SIMILARITIES)}, got '{train_config[TrainingKeys.SIMILARITY]}'"
                )

    def _validate_datasets(self) -> None:
        for idx, ds_config in enumerate(self.config[ConfigSections.DATASETS]):
            for key in REQUIRED_DATASET_KEYS:
                if key not in ds_config:
                    raise ValueError(
                        f"Dataset {idx} ('{ds_config.get(DatasetKeys.NAME, '?')}'): missing required key '{key}'"
                    )
            unknown = set(ds_config.keys()) - VALID_DATASET_KEYS
            if unknown:
                logger.warning(
                    f"Dataset '{ds_config.get(DatasetKeys.NAME, '?')}': "
                    f"unrecognized keys (possible typos): {sorted(unknown)}."
                )
            dataset_config = DatasetConfig.from_dict(ds_config)
            if dataset_config.weight <= 0:
                raise ValueError(f"Dataset '{dataset_config.name}': '{DatasetKeys.WEIGHT}' must be positive")
            if dataset_config.type not in (DatasetTypes.UNPAIRED, DatasetTypes.PAIRED):
                raise ValueError(
                    f"Dataset '{dataset_config.name}': unknown type '{dataset_config.type}'. "
                    f"Must be '{DatasetTypes.UNPAIRED}' or '{DatasetTypes.PAIRED}'."
                )


def validate_config(config: Dict[str, Any]) -> None:
    ConfigValidator(config).validate()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _resolve_path(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


class DatasetJsonCache:
    """Memoizes JSON parsing + path resolution across the multiple validators
    and loaders that consume the same dataset files in one build."""

    def __init__(self) -> None:
        self._content_cache: Dict[str, Dict[str, Any]] = {}
        self._entry_cache: Dict[str, List[Dict[str, str]]] = {}

    def load_content(self, json_path: str) -> Dict[str, Any]:
        json_path = os.path.abspath(json_path)
        if json_path not in self._content_cache:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON dataset file not found: {json_path}")
            with open(json_path, 'r') as f:
                content = json.load(f)
            if JsonKeys.DATA not in content:
                raise ValueError(f"JSON dataset file {json_path} must contain top-level '{JsonKeys.DATA}' key.")
            self._content_cache[json_path] = content
        return self._content_cache[json_path]

    def load_entries(self, json_path: str) -> List[Dict[str, str]]:
        json_path = os.path.abspath(json_path)
        if json_path not in self._entry_cache:
            self._entry_cache[json_path] = self._load_and_resolve_entries(json_path)
        return [dict(item) for item in self._entry_cache[json_path]]

    def _load_and_resolve_entries(self, json_path: str) -> List[Dict[str, str]]:
        content = self.load_content(json_path)
        data = [dict(item) for item in content[JsonKeys.DATA]]
        base_dir = os.path.dirname(os.path.abspath(json_path))

        for idx, item in enumerate(data):
            if dataset.Fields.IMAGE not in item:
                raise ValueError(f"Missing '{dataset.Fields.IMAGE}' field in entry {idx} of {json_path}")

            image_path = item[dataset.Fields.IMAGE]
            resolved_image = _resolve_path(image_path, base_dir)
            if not os.path.exists(resolved_image):
                raise FileNotFoundError(
                    f"Image file not found for entry {idx} in {json_path}: "
                    f"{image_path} (resolved to {resolved_image})"
                )
            item[dataset.Fields.IMAGE] = resolved_image

            for field in OPTIONAL_DATA_FIELDS:
                # Truthy check so ``{"segmentation": null}`` is treated as absent.
                path = item.get(field)
                if not path:
                    item.pop(field, None)
                    continue
                resolved = _resolve_path(path, base_dir)
                if not os.path.exists(resolved):
                    raise FileNotFoundError(
                        f"{field.capitalize()} file not found for entry {idx} in {json_path}: "
                        f"{path} (resolved to {resolved})"
                    )
                item[field] = resolved

        return data

    def entry_field_sets(self, json_path: str) -> List[FrozenSet[str]]:
        content = self.load_content(json_path)
        if not content[JsonKeys.DATA]:
            raise ValueError(f"JSON file {json_path} has no data entries")
        return [
            frozenset(k for k in OPTIONAL_DATA_FIELDS if k in entry)
            for entry in content[JsonKeys.DATA]
        ]


def required_data_fields(training: TrainingConfig) -> FrozenSet[str]:
    required = set()
    if training.dice_loss_weight > 0.0:
        required.add(dataset.Fields.SEGMENTATION)
    if training.loss_function_masking or training.roi_masking:
        required.add(dataset.Fields.MASK)
    return frozenset(required)


def determine_data_fields(
    dataset_configs: List["DatasetConfig"],
    config_dir: str,
    json_cache: Optional[DatasetJsonCache] = None,
) -> Dict[str, List[FrozenSet[str]]]:
    json_cache = json_cache or DatasetJsonCache()
    fields_per_dataset = {}

    for ds_config in dataset_configs:
        json_file = _resolve_path(ds_config.json_file, config_dir)
        fields_per_dataset[ds_config.name] = json_cache.entry_field_sets(json_file)

    return fields_per_dataset


def validate_paired_datasets_have_pairs(
    schema: "FinetuningConfigSchema",
    config_dir: str,
    json_cache: DatasetJsonCache,
) -> None:
    """Catches missing-pairs at config-validation time so we don't spend
    minutes preprocessing images before ``SubjectPairSampler`` raises."""
    for ds in schema.datasets:
        if ds.type != DatasetTypes.PAIRED:
            continue
        json_file = _resolve_path(ds.json_file, config_dir)
        content = json_cache.load_content(json_file)
        subject_counts: Dict[str, int] = {}
        for entry in content[JsonKeys.DATA]:
            sid = entry.get(dataset.Fields.SUBJECT_ID)
            if sid:
                subject_counts[sid] = subject_counts.get(sid, 0) + 1
        if not subject_counts:
            raise ValueError(
                f"Dataset '{ds.name}': type is 'paired' but no entry has a "
                f"'subject_id' field. Add 'subject_id' to each entry, or change "
                f"the dataset type to 'unpaired'."
            )
        if not any(count >= 2 for count in subject_counts.values()):
            raise ValueError(
                f"Dataset '{ds.name}': type is 'paired' but no subject has "
                f"two or more entries (paired sampling requires at least one "
                f"subject with >=2 images sharing the same 'subject_id'). "
                f"Found {len(subject_counts)} subject(s)."
            )


def validate_training_data_compatibility(
    fields_per_dataset: Dict[str, List[FrozenSet[str]]],
    data_fields: FrozenSet[str],
) -> None:
    for dataset_name, entry_fields in fields_per_dataset.items():
        for idx, present_fields in enumerate(entry_fields):
            missing = data_fields - present_fields
            if missing:
                raise ValueError(
                    f"Dataset '{dataset_name}' entry {idx} is missing required field(s) "
                    f"{sorted(missing)} for the current training config."
                )

    available_fields = frozenset().union(
        *(present_fields for entry_fields in fields_per_dataset.values() for present_fields in entry_fields)
    ) if fields_per_dataset else frozenset()
    ignored_fields = available_fields - data_fields
    if ignored_fields:
        logger.warning(
            f"Ignoring auxiliary JSON field(s) {sorted(ignored_fields)} that are present in the "
            f"data but not required by the current training configuration."
        )


def _keep_required_optional_fields(data: List[Dict[str, str]], data_fields: FrozenSet[str]) -> List[Dict[str, str]]:
    keep_fields = OPTIONAL_DATA_FIELDS & data_fields
    filtered = []
    for item in data:
        filtered_item = dict(item)
        for field in OPTIONAL_DATA_FIELDS - keep_fields:
            filtered_item.pop(field, None)
        filtered.append(filtered_item)
    return filtered


def create_dataset_from_config(dataset_config: "DatasetConfig", input_shape: Tuple[int, ...],
                               config_dir: str = "", use_label: bool = False,
                               data_fields: FrozenSet[str] = frozenset(),
                               json_cache: Optional[DatasetJsonCache] = None) -> dataset.Dataset:
    """Build a ``Dataset`` (random pairing) or ``PairedDataset`` (subject-based
    pairing) based on ``dataset_config.type``. Auxiliary data (segmentations,
    masks) is filtered by ``data_fields``, not by the dataset type."""
    json_cache = json_cache or DatasetJsonCache()

    json_file = _resolve_path(dataset_config.json_file, config_dir)
    # Spread DatasetParams' fields so adding an optional parameter to
    # ``DatasetParams`` flows through without touching this mapping.
    common_params = {
        'input_shape': input_shape,
        'name': dataset_config.name,
        'data': _keep_required_optional_fields(json_cache.load_entries(json_file), data_fields),
        'use_label': use_label,
        **{f.name: getattr(dataset_config, f.name) for f in fields(dataset.DatasetParams)},
    }

    if dataset_config.type == DatasetTypes.UNPAIRED:
        return dataset.Dataset(**common_params)
    elif dataset_config.type == DatasetTypes.PAIRED:
        return dataset.PairedDataset(**common_params)
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_config.type}. "
            f"Must be '{DatasetTypes.UNPAIRED}' or '{DatasetTypes.PAIRED}'."
        )


def _build_datasets_and_val_loaders(
    dataset_configs: List["DatasetConfig"],
    input_shape: Tuple[int, ...],
    config_dir: str,
    use_label: bool,
    data_fields: FrozenSet[str],
    json_cache: DatasetJsonCache,
) -> Tuple[List[dataset.Dataset], List[float], Dict[str, DataLoader]]:
    datasets: List[dataset.Dataset] = []
    weights: List[float] = []
    val_loaders: Dict[str, DataLoader] = {}

    logger.info(f"Building {len(dataset_configs)} dataset(s).")
    for ds_config in dataset_configs:
        logger.info(
            f"Building dataset '{ds_config.name}' "
            f"(type={ds_config.type}, sampling_weight={ds_config.weight})."
        )

        ds = create_dataset_from_config(
            ds_config,
            input_shape,
            config_dir=config_dir,
            use_label=use_label,
            data_fields=data_fields,
            json_cache=json_cache,
        )
        datasets.append(ds)

        ds_length = len(ds)
        weights.extend([ds_config.weight / ds_length] * ds_length)

        logger.info(f"Dataset '{ds_config.name}': {ds_length} samples available.")
        # ``shuffle=True`` so each validation call sees a different anchor;
        # ``num_workers=0`` avoids re-spawning workers per call.
        val_loaders[ds_config.name] = DataLoader(
            ds,
            batch_size=DEFAULT_VAL_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            drop_last=DEFAULT_DROP_LAST,
            pin_memory=DEFAULT_PIN_MEMORY,
        )

    return datasets, weights, val_loaders


def _prepare_config(config_path: str, config: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], str, FinetuningConfigSchema]:
    if config is None:
        config = load_config(config_path)
    validate_config(config)

    prepared_config = copy.deepcopy(config)
    schema = FinetuningConfigSchema.from_dict(prepared_config)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    return prepared_config, config_dir, schema


def _validate_data_requirements(
    schema: FinetuningConfigSchema,
    config_dir: str,
    json_cache: DatasetJsonCache,
) -> FrozenSet[str]:
    data_fields = required_data_fields(schema.training)
    validate_paired_datasets_have_pairs(schema, config_dir, json_cache)
    fields_per_dataset = determine_data_fields(schema.datasets, config_dir, json_cache)
    validate_training_data_compatibility(fields_per_dataset, data_fields)
    logger.info(
        f"Required auxiliary data fields: "
        f"{sorted(data_fields) if data_fields else 'images only'}."
    )
    return data_fields


def _create_train_loader(
    datasets: List[dataset.Dataset],
    weights: List[float],
    training: TrainingConfig,
) -> Tuple[DataLoader, int, int, int]:
    combined_dataset = ConcatDataset(datasets)
    total_samples = len(combined_dataset)
    samples_per_epoch = training.samples_per_epoch or total_samples
    num_gpus = len(training.gpus)
    effective_batch_size = training.batch_size * num_gpus
    iterations_per_epoch = samples_per_epoch // effective_batch_size
    if iterations_per_epoch == 0:
        raise ValueError(
            f"samples_per_epoch ({samples_per_epoch}) is smaller than the "
            f"effective batch size ({training.batch_size} x {num_gpus} GPU(s) "
            f"= {effective_batch_size}); each epoch would yield zero training "
            f"iterations. Increase samples_per_epoch or decrease batch_size."
        )
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    train_loader = DataLoader(
        combined_dataset,
        batch_size=effective_batch_size,
        num_workers=training.num_workers,
        drop_last=DEFAULT_DROP_LAST,
        pin_memory=DEFAULT_PIN_MEMORY,
        worker_init_fn=_seed_worker,
        sampler=WeightedRandomSampler(
            weights=normalized_weights,
            num_samples=samples_per_epoch,
            replacement=DEFAULT_SAMPLER_REPLACEMENT,
        ),
    )

    return train_loader, total_samples, samples_per_epoch, iterations_per_epoch


def create_data_loaders(config_path: str, config: Optional[Dict[str, Any]] = None) -> DataLoaderBundle:
    """
    Create training and validation dataloaders from YAML config.

    Args:
        config_path: Path to YAML config file (used to resolve relative paths)
        config: Pre-loaded config dict. If None, loads from config_path.

    Returns:
        DataLoaderBundle containing training loader, validation loaders,
        defaulted config dictionary, and required auxiliary data fields.
    """
    config, config_dir, schema = _prepare_config(config_path, config)
    json_cache = DatasetJsonCache()
    data_fields = _validate_data_requirements(schema, config_dir, json_cache)

    datasets, weights, val_loaders = _build_datasets_and_val_loaders(
        schema.datasets,
        tuple(schema.training.input_shape),
        config_dir,
        schema.training.use_label,
        data_fields,
        json_cache,
    )

    train_loader, total_samples, samples_per_epoch, iterations_per_epoch = _create_train_loader(
        datasets, weights, schema.training,
    )

    logger.info(
        f"Training loader ready: total_samples={total_samples}, "
        f"samples_per_epoch={samples_per_epoch}, "
        f"effective_batch={schema.training.batch_size}x{len(schema.training.gpus)} GPU(s), "
        f"iterations_per_epoch={iterations_per_epoch}."
    )

    return DataLoaderBundle(
        train_loader=train_loader,
        val_loaders=val_loaders,
        config=config,
        data_fields=data_fields,
    )
