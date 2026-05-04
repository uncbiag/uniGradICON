import logging
import torch
import numpy as np
import collections
import hashlib
import json
from tqdm import tqdm
import random
import os
import footsteps
import itk
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from torch.utils.data import Dataset as TorchDataset

import blosc
blosc.set_nthreads(1)

logger = logging.getLogger(__name__)


class Fields:
    IMAGE = "image"
    SEGMENTATION = "segmentation"
    MASK = "mask"
    SUBJECT_ID = "subject_id"
    MODALITY = "modality"


class CacheNames:
    IMAGES = "images"
    SEGMENTATIONS = "segmentations"
    MASKS = "masks"


_AUX_CACHE_NAMES = {
    Fields.SEGMENTATION: CacheNames.SEGMENTATIONS,
    Fields.MASK: CacheNames.MASKS,
}


class PairKeys:
    IMAGE_A = "image_A"
    IMAGE_B = "image_B"
    SEGMENTATION_A = "segmentation_A"
    SEGMENTATION_B = "segmentation_B"
    MASK_A = "mask_A"
    MASK_B = "mask_B"
    LABEL_A = "label_A"
    LABEL_B = "label_B"


@dataclass
class DatasetParams:
    """Shared defaults for ``Dataset.__init__`` (via ``_DEFAULTS``) and
    ``DatasetConfig`` (via inheritance) so a new optional parameter is
    declared in exactly one place."""
    cache_dir: Optional[str] = None
    maximum_images: Optional[int] = None
    use_cache: bool = True
    use_compression: bool = False
    is_ct: bool = False
    ct_window: Tuple[float, float] = (-1000, 1000)
    quantile_range: Tuple[float, float] = (0.0, 0.99)
    shuffle: bool = True


_DEFAULTS = DatasetParams()


def _stable_hash(value) -> str:
    """SHA-256 of a JSON dump. ``hash()`` is salted per-process so we can't
    use it for cache keys."""
    if not value:
        return ""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reorient(moving: "itk.Image") -> "itk.Image":
    desired_coordinate_orientation = itk.ITKCommonBasePython.itkSpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_RAS
    if hasattr(itk, "AnatomicalOrientation"):
        desired_coordinate_orientation = itk.AnatomicalOrientation(desired_coordinate_orientation)

    return itk.orient_image_filter(
        moving,
        desired_coordinate_orientation=desired_coordinate_orientation,
        use_image_direction=True)


@dataclass
class DatasetEntry:
    image: str
    segmentation: Optional[str] = None
    mask: Optional[str] = None
    subject_id: Optional[str] = None
    modality: Optional[str] = None

    @classmethod
    def from_dict(cls, item: Dict[str, str], idx: int, dataset_name: str) -> "DatasetEntry":
        if Fields.IMAGE not in item:
            raise ValueError(f"Dataset {dataset_name}: missing '{Fields.IMAGE}' field in entry {idx}")
        return cls(
            image=item[Fields.IMAGE],
            segmentation=item.get(Fields.SEGMENTATION),
            mask=item.get(Fields.MASK),
            subject_id=item.get(Fields.SUBJECT_ID),
            modality=item.get(Fields.MODALITY),
        )


def _build_required_field_map(entries: Sequence[DatasetEntry], field_name: str, dataset_name: str) -> Dict[str, str]:
    """Image-to-field map. Field must be present on every entry or none."""
    entries_with_field = [entry for entry in entries if getattr(entry, field_name) is not None]
    if not entries_with_field:
        return {}

    missing = [entry.image for entry in entries if getattr(entry, field_name) is None]
    if missing:
        raise ValueError(
            f"Dataset '{dataset_name}' has '{field_name}' for {len(entries_with_field)} of {len(entries)} "
            f"entries. Either provide '{field_name}' for every entry or remove it from every entry. "
            f"First missing image: {missing[0]}"
        )

    return {entry.image: getattr(entry, field_name) for entry in entries}


class ImageReader:
    def read(self, path: str) -> torch.Tensor:
        itk_image = _reorient(itk.imread(path, itk.F))
        # GetArrayFromImage returns an owned numpy array (unlike the *View*
        # variant), so from_numpy is safe and skips the extra tensor copy.
        image = itk.GetArrayFromImage(itk_image)
        return torch.from_numpy(image)


class ImagePreprocessor:
    def __init__(
        self,
        reader: ImageReader,
        input_shape: Tuple[int, ...],
        is_ct: bool,
        ct_window: Tuple[float, float],
        quantile_range: Tuple[float, float],
        modality_map: Dict[str, bool],
    ):
        self.reader = reader
        self.input_shape = input_shape
        self.is_ct = is_ct
        self.ct_window = ct_window
        self.quantile_range = quantile_range
        self.modality_map = modality_map

    def preprocess_image(self, path: str) -> torch.Tensor:
        volume = self.reader.read(path)
        volume = volume[None, None].float()
        volume = torch.nn.functional.interpolate(
            volume, self.input_shape, mode="trilinear", align_corners=True
        )

        is_ct = self.modality_map.get(path, self.is_ct)
        if is_ct:
            im_min, im_max = self.ct_window
        else:
            flat = volume.view(-1).numpy()
            im_min = float(np.quantile(flat, self.quantile_range[0]))
            im_max = float(np.quantile(flat, self.quantile_range[1]))

        normalized = torch.clamp(volume, im_min, im_max)
        normalized = normalized - im_min
        intensity_range = im_max - im_min
        if intensity_range > 0:
            normalized = normalized / intensity_range

        return normalized[0]

    def preprocess_label_map(self, path: str) -> torch.Tensor:
        label_map = self.reader.read(path)
        label_map = label_map[None, None].float()
        label_map = torch.nn.functional.interpolate(label_map, self.input_shape, mode="nearest")
        return label_map[0]


class DatasetCache:
    """Cache path is partitioned by a signature hashed from every
    preprocessing-affecting parameter, so configs that share a ``cache_dir``
    but differ in any of those parameters get distinct cache files (no
    thrashing) and identical configs share the cache transparently."""

    def __init__(
        self,
        dataset_name: str,
        cache_dir: Optional[str],
        enabled: bool,
        signature: str,
    ):
        self.dataset_name = dataset_name
        self.enabled = enabled
        self.signature = signature
        if not enabled:
            self.base_dir = None
        else:
            root = cache_dir if cache_dir else footsteps.output_dir
            if root is None:
                raise ValueError(
                    f"Dataset '{dataset_name}': caching is enabled but no cache_dir was "
                    f"provided and footsteps.output_dir is unset. "
                    f"Call footsteps.initialize() before constructing the dataset, "
                    f"or pass cache_dir explicitly, or set use_cache=False."
                )
            self.base_dir = os.path.join(root, signature)

    def path(self, cache_name: str) -> Optional[str]:
        if not self.enabled:
            return None
        return os.path.join(self.base_dir, f"{self.dataset_name}_cached_{cache_name}.trch")

    def load(self, cache_name: str) -> Optional[dict]:
        cache_path = self.path(cache_name)
        if not (cache_path and os.path.exists(cache_path)):
            return None
        try:
            return torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception as e:
            # Treat any deserialization failure as "no cache" — caller rebuilds.
            logger.warning(
                f"Dataset '{self.dataset_name}': failed to deserialize cache file "
                f"{cache_path} ({e}); rebuilding from source."
            )
            return None

    def save(self, cache_name: str, payload):
        cache_path = self.path(cache_name)
        if not cache_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        # Atomic write: tmp file + rename so concurrent readers never see a
        # partial file.
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)

    def write_metadata(self, meta: Dict[str, Any]) -> None:
        """Sidecar ``_meta.json`` describing the params behind the
        hash-named directory. Skipped when the file already exists since the
        signature uniquely determines the content."""
        if not self.enabled:
            return
        meta_path = os.path.join(self.base_dir, "_meta.json")
        if os.path.exists(meta_path):
            return
        os.makedirs(self.base_dir, exist_ok=True)
        tmp_path = f"{meta_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True, default=str)
        os.replace(tmp_path, meta_path)


class RandomPairSampler:
    def __init__(self, keys: Sequence[str]):
        self.keys = list(keys)
        if len(self.keys) < 2:
            raise ValueError("RandomPairSampler needs at least 2 keys")
        self._key_to_index = {k: i for i, k in enumerate(self.keys)}

    def sample_partner(self, anchor: str) -> str:
        # Pick uniformly from the n-1 keys that are not ``anchor`` without
        # rejection-sampling: draw an index in [0, n-1) and bump past anchor.
        anchor_idx = self._key_to_index[anchor]
        partner_idx = random.randrange(len(self.keys) - 1)
        if partner_idx >= anchor_idx:
            partner_idx += 1
        return self.keys[partner_idx]


class SubjectPairSampler:
    """Pair only images sharing the same ``subject_id``."""

    def __init__(self, entries: Sequence[DatasetEntry], keys: Sequence[str], dataset_name: str):
        subject_lookup = collections.defaultdict(list)
        path_to_subject = {}
        valid_keys = set(keys)
        for entry in entries:
            if entry.image in valid_keys and entry.subject_id:
                path_to_subject[entry.image] = entry.subject_id
                subject_lookup[entry.subject_id].append(entry.image)

        self.pair_candidates = {}
        for path, subject_id in path_to_subject.items():
            others = [key for key in subject_lookup[subject_id] if key != path]
            if others:
                self.pair_candidates[path] = others

        self.keys = [key for key in keys if key in self.pair_candidates]
        if not self.keys:
            raise ValueError(
                f"Dataset '{dataset_name}': no valid pairs found. "
                f"Ensure data entries have 'subject_id' and at least 2 images per subject."
            )

    def sample_partner(self, anchor: str) -> str:
        return random.choice(self.pair_candidates[anchor])


class Dataset(TorchDataset):
    """3D medical-image registration dataset.

    ``__getitem__(index)`` selects the anchor image (image_A) by index and
    samples a partner (image_B) randomly from the dataset. This makes the
    weighting in ``WeightedRandomSampler`` and the partitioning in
    ``DistributedSampler`` actually drive what's loaded; pair construction
    keeps its randomness via the partner draw.

    When ``use_compression=True``, tensors are compressed with blosc at
    load time and decompressed lazily inside ``_build_pair`` so DataLoader
    workers each pay the per-sample decompression cost in parallel.
    """

    def __init__(self,
                 input_shape: Tuple[int, ...],
                 name: str,
                 data: List[Dict[str, str]],
                 cache_dir: Optional[str] = _DEFAULTS.cache_dir,
                 maximum_images: Optional[int] = _DEFAULTS.maximum_images,
                 shuffle: bool = _DEFAULTS.shuffle,
                 is_ct: bool = _DEFAULTS.is_ct,
                 ct_window: Tuple[float, float] = _DEFAULTS.ct_window,
                 quantile_range: Tuple[float, float] = _DEFAULTS.quantile_range,
                 use_cache: bool = _DEFAULTS.use_cache,
                 use_compression: bool = _DEFAULTS.use_compression,
                 use_label: bool = False):

        self.name = name
        self.input_shape = tuple(input_shape)
        self.is_ct = is_ct
        self.ct_window = tuple(ct_window)
        self.quantile_range = tuple(quantile_range)
        self.use_compression = use_compression
        self.use_label = use_label

        self._build_entries_and_field_maps(data)
        self._build_preprocessor()
        self._initialize_cache(cache_dir, use_cache, maximum_images)

        loaded_paths = self._load_image_store(maximum_images, shuffle)
        if len(loaded_paths) < 2:
            raise ValueError(
                f"Dataset '{self.name}': at least 2 images are required to form "
                f"registration pairs, but only {len(loaded_paths)} loaded successfully."
            )
        logger.info(f"Dataset '{self.name}': {len(loaded_paths)} image(s) loaded.")

        # Build the pair sampler before subject maps and auxiliary maps so
        # both index the post-filter ``self.keys`` (PairedDataset drops
        # orphan-subject images here).
        self.keys = loaded_paths
        self.pair_sampler = self._create_pair_sampler()
        self.keys = self.pair_sampler.keys

        self._build_subject_maps()

        if self.has_segmentation:
            self._load_label_maps(Fields.SEGMENTATION, self._segmentation_map)
        if self.has_mask:
            self._load_label_maps(Fields.MASK, self._mask_map)

    def _build_entries_and_field_maps(self, data: List[Dict[str, str]]) -> None:
        if not data:
            raise ValueError(f"Dataset {self.name}: 'data' must be provided (from JSON source)")
        self._data_fingerprint = _stable_hash(data)
        self.entries = [
            DatasetEntry.from_dict(item, idx, self.name)
            for idx, item in enumerate(data)
        ]
        self._segmentation_map = _build_required_field_map(self.entries, Fields.SEGMENTATION, self.name)
        self._mask_map = _build_required_field_map(self.entries, Fields.MASK, self.name)

        self._modality_map: Dict[str, bool] = {}
        for entry in self.entries:
            if entry.modality is not None:
                self._modality_map[entry.image] = entry.modality.lower() == 'ct'
        self._modality_hash = _stable_hash(self._modality_map)

        if self._modality_map and len(self._modality_map) < len(data):
            missing = [entry.image for entry in self.entries if entry.image not in self._modality_map]
            fallback = "CT" if self.is_ct else "MRI"
            logger.warning(
                f"Dataset '{self.name}': {len(missing)} of {len(data)} entries lack a 'modality' field "
                f"and will fall back to the dataset-level is_ct={self.is_ct} ({fallback} preprocessing). "
                f"First entry without modality: {missing[0]}."
            )

    def _build_preprocessor(self) -> None:
        self.reader = ImageReader()
        self.preprocessor = ImagePreprocessor(
            self.reader,
            self.input_shape,
            self.is_ct,
            self.ct_window,
            self.quantile_range,
            self._modality_map,
        )

    def _initialize_cache(
        self,
        cache_dir: Optional[str],
        use_cache: bool,
        maximum_images: Optional[int],
    ) -> None:
        cache_params = {
            "dataset_name": self.name,
            "input_shape": list(self.input_shape),
            "is_ct": self.is_ct,
            "ct_window": list(self.ct_window),
            "quantile_range": list(self.quantile_range),
            "modality_hash": self._modality_hash,
            "maximum_images": maximum_images,
            "use_compression": self.use_compression,
            "data_fingerprint": self._data_fingerprint,
        }
        cache_signature = _stable_hash(cache_params)
        self.cache = DatasetCache(self.name, cache_dir, use_cache, cache_signature)
        self.cache.write_metadata({"signature": cache_signature, **cache_params})

    def _load_image_store(self, maximum_images: Optional[int], shuffle: bool) -> List[str]:
        # Sort before slicing so the ``maximum_images`` subset is deterministic
        # across runs; otherwise ``shuffle=True`` picks a different subset every
        # run and the cache thrashes.
        paths = sorted(entry.image for entry in self.entries)
        if maximum_images is not None:
            paths = paths[:maximum_images]
        if shuffle:
            random.shuffle(paths)

        loaded_cache = self.cache.load(CacheNames.IMAGES)
        should_save_cache = loaded_cache is None
        self.store = loaded_cache if loaded_cache is not None else {}

        missing_paths = [path for path in paths if path not in self.store]
        if missing_paths:
            self._load_images(missing_paths)
            should_save_cache = True

        if len(self.store) != len(paths):
            should_save_cache = True
        self.store = {path: self.store[path] for path in paths if path in self.store}

        if self.cache.enabled and should_save_cache:
            self.cache.save(CacheNames.IMAGES, self.store)

        return list(self.store.keys())

    def _build_subject_maps(self) -> None:
        key_set = set(self.keys)
        self._subject_modality_images: Dict[str, Dict[str, List[str]]] = {}
        self._key_to_subject = {}
        for entry in self.entries:
            if entry.subject_id and entry.image in key_set:
                mod = (entry.modality or 'default').lower()
                self._subject_modality_images.setdefault(entry.subject_id, {}).setdefault(mod, []).append(entry.image)
                self._key_to_subject[entry.image] = entry.subject_id

        if self.use_label and not self._key_to_subject:
            logger.warning(
                f"Dataset '{self.name}': use_label is enabled but no entries carry "
                f"'subject_id', so labels will be copies of the input images (a no-op). "
                f"Add 'subject_id' to the JSON or remove use_label."
            )

    def _create_pair_sampler(self):
        return RandomPairSampler(self.keys)

    def _load_label_maps(self, field_name: str, path_map: Dict[str, str]):
        cache_name = _AUX_CACHE_NAMES[field_name]
        cache = self.cache.load(cache_name)
        # Subset check guards against a stale auxiliary cache when self.keys
        # is now a superset (e.g. an image that previously failed to load now
        # succeeds, so its label was never written to the cache).
        if cache is not None and set(self.keys).issubset(cache):
            for path in self.keys:
                self.store[path][field_name] = cache[path]
            return

        failures = []
        for path in tqdm(self.keys, desc=f"Loading {field_name} maps for '{self.name}'"):
            label_path = path_map.get(path)
            if label_path is None:
                failures.append((path, f"missing '{field_name}' path"))
                continue
            try:
                self.store[path][field_name] = self._compress(self.preprocessor.preprocess_label_map(label_path))
            except Exception as e:
                logger.warning(
                    f"Dataset '{self.name}': failed to preprocess {field_name} map "
                    f"for image '{path}': {e}."
                )
                failures.append((path, e))

        if failures:
            raise RuntimeError(
                f"Dataset '{self.name}': failed to load {len(failures)}/{len(self.keys)} "
                f"{field_name} map(s). Auxiliary cache was not written; fix the input data and retry. "
                f"First failure: {failures[0][0]}: {failures[0][1]}"
            )

        self.cache.save(
            cache_name,
            {path: self.store[path][field_name] for path in self.keys},
        )

    def _load_images(self, paths: List[str]):
        cached_count = len(self.store)
        failures = []
        for path in tqdm(paths, desc=f"Loading images for '{self.name}'"):
            try:
                self.store[path] = {Fields.IMAGE: self._compress(self.preprocessor.preprocess_image(path))}
            except Exception as e:
                logger.warning(
                    f"Dataset '{self.name}': failed to load image '{path}': {e}."
                )
                failures.append((path, e))

        if failures:
            failed = len(failures)
            total = len(paths)
            if failed == total and cached_count == 0:
                raise RuntimeError(
                    f"Dataset '{self.name}': all {failed} requested image(s) failed to load. "
                    f"First failure: '{failures[0][0]}': {failures[0][1]}."
                )
            pct = failed / total * 100
            logger.warning(
                f"Dataset '{self.name}': {failed}/{total} ({pct:.1f}%) images failed to load. "
                f"Failed entries were not added to the cache and will be retried on the next run."
            )

    @property
    def has_segmentation(self) -> bool:
        return bool(self._segmentation_map)

    @property
    def has_mask(self) -> bool:
        return bool(self._mask_map)

    def _compress(self, tensor: torch.Tensor) -> Union[torch.Tensor, bytes]:
        if self.use_compression:
            return blosc.pack_array(tensor.detach().cpu().contiguous().numpy())
        return tensor

    def _decompress(self, packed: Union[torch.Tensor, bytes]) -> torch.Tensor:
        if isinstance(packed, bytes):
            return torch.from_numpy(blosc.unpack_array(packed))
        return packed

    def get_image(self, key: str) -> torch.Tensor:
        return self._decompress(self.store[key][Fields.IMAGE])

    def _build_pair(self, key_a: str, key_b: str) -> Dict[str, torch.Tensor]:
        result = {
            PairKeys.IMAGE_A: self.get_image(key_a),
            PairKeys.IMAGE_B: self.get_image(key_b),
        }
        if self.has_segmentation:
            result[PairKeys.SEGMENTATION_A] = self._decompress(self.store[key_a][Fields.SEGMENTATION])
            result[PairKeys.SEGMENTATION_B] = self._decompress(self.store[key_b][Fields.SEGMENTATION])
        if self.has_mask:
            result[PairKeys.MASK_A] = self._decompress(self.store[key_a][Fields.MASK])
            result[PairKeys.MASK_B] = self._decompress(self.store[key_b][Fields.MASK])
        if self.use_label:
            self._add_label_images(result, key_a, key_b)
        return result

    def _add_label_images(self, result: Dict[str, torch.Tensor], key_a: str, key_b: str) -> None:
        subject_a = self._key_to_subject.get(key_a)
        subject_b = self._key_to_subject.get(key_b)
        modality = (self._sample_common_label_modality(subject_a, subject_b)
                    if subject_a and subject_b else None)
        if modality is None:
            result[PairKeys.LABEL_A] = result[PairKeys.IMAGE_A]
            result[PairKeys.LABEL_B] = result[PairKeys.IMAGE_B]
            return
        result[PairKeys.LABEL_A] = self.get_image(random.choice(self._subject_modality_images[subject_a][modality]))
        result[PairKeys.LABEL_B] = self.get_image(random.choice(self._subject_modality_images[subject_b][modality]))

    def _sample_common_label_modality(self, subject_a: str, subject_b: str) -> Optional[str]:
        mods_a = set(self._subject_modality_images[subject_a].keys())
        mods_b = set(self._subject_modality_images[subject_b].keys())
        common = sorted(mods_a & mods_b)
        if not common:
            return None
        return random.choice(common)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        key_a = self.keys[index]
        key_b = self.pair_sampler.sample_partner(key_a)
        return self._build_pair(key_a, key_b)


class PairedDataset(Dataset):
    """Variant of ``Dataset`` that pairs only within ``subject_id``. JSON
    entries must include ``subject_id`` and at least one subject must have
    ≥2 images."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(
            f"PairedDataset '{self.name}': "
            f"{len(self.keys)} image(s) form valid same-subject pairs."
        )

    def _create_pair_sampler(self):
        return SubjectPairSampler(self.entries, self.keys, self.name)
