import logging
import torch
import numpy as np
import collections
from tqdm import tqdm
import random
import os
import footsteps
import itk
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset as TorchDataset

try:
    import blosc
    blosc.set_nthreads(1)
    _HAS_BLOSC = True
except ImportError:
    _HAS_BLOSC = False

logger = logging.getLogger(__name__)


def _deterministic_hash(modality_map: dict) -> str:
    """Compute a deterministic hash of the modality map for cache validation.
    Python's built-in hash() is randomized across processes (PYTHONHASHSEED),
    so we use a sorted string representation instead."""
    if not modality_map:
        return ""
    return str(sorted(modality_map.items()))


def reorient(moving):
    desired_coordinate_orientation = itk.ITKCommonBasePython.itkSpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_RAS

    if hasattr(itk, "AnatomicalOrientation"):
        desired_coordinate_orientation = itk.AnatomicalOrientation(desired_coordinate_orientation)

    return itk.orient_image_filter(
        moving,
        desired_coordinate_orientation=desired_coordinate_orientation,
        use_image_direction=True)


def _validate_cache(cache: dict, name: str, maximum_images, read_type: str, is_ct: bool,
                    ct_window: Tuple[float, float], quantile_range: Tuple[float, float],
                    modality_hash: str = ""):
    """Validate cache metadata matches current dataset parameters."""
    errors = []
    if cache.get("name") != name:
        errors.append(f"name: expected '{name}', got '{cache.get('name')}'")
    if cache.get("maximum_images") != maximum_images:
        errors.append(f"maximum_images: expected {maximum_images}, got {cache.get('maximum_images')}")
    if cache.get("read_type") != read_type:
        errors.append(f"read_type: expected '{read_type}', got '{cache.get('read_type')}'")
    if cache.get("is_ct") != is_ct:
        errors.append(f"is_ct: expected {is_ct}, got {cache.get('is_ct')}")
    if cache.get("ct_window") != ct_window:
        errors.append(f"ct_window: expected {ct_window}, got {cache.get('ct_window')}")
    if cache.get("quantile_range") != quantile_range:
        errors.append(f"quantile_range: expected {quantile_range}, got {cache.get('quantile_range')}")
    if cache.get("modality_hash") != modality_hash:
        errors.append(f"per-image modality settings changed")
    if errors:
        raise ValueError(
            f"Cache file is stale or incompatible with current config. Mismatches: {'; '.join(errors)}. "
            f"Delete the cache file and retry."
        )


def _build_pair_lookup(data, store, keys):
    """Build subject-based pair lookup from loaded data.
    Returns (pair_candidates, filtered_keys) where pair_candidates maps each
    image path to a list of other paths from the same subject."""
    subject_lookup = collections.defaultdict(list)
    path_to_subject = {}
    for item in data:
        path = item['image']
        subject_id = item.get('subject_id')
        if path in store and subject_id:
            path_to_subject[path] = subject_id
            subject_lookup[subject_id].append(path)

    pair_candidates = {}
    for path, subject_id in path_to_subject.items():
        others = [k for k in subject_lookup[subject_id] if k != path]
        if others:
            pair_candidates[path] = others

    filtered_keys = [k for k in keys if k in pair_candidates]
    return pair_candidates, filtered_keys


class Dataset(TorchDataset):
    """Dataset for medical image registration.

    Loads and preprocesses 3D medical images for registration training.
    Optionally loads segmentation maps and/or binary masks based on the
    fields present in the JSON data entries:

    - ``segmentation``: integer label maps for Dice loss computation
    - ``mask``: binary ROI masks for loss function masking and/or image cropping

    Supports two image readers via ``read_type``:
    - ``"itk"`` (default): reads NIfTI, NRRD, and other ITK-supported formats.
    - ``"dicom"``: reads DICOM series directories (pass the directory path as ``image``).

    Note: __getitem__ returns random image pairs regardless of the index argument.
    This is by design for registration training where random pairing is standard.
    """

    def __init__(self,
                 input_shape: Tuple[int, ...],
                 name: str,
                 data: List[Dict[str, str]],
                 read_type: str = "itk",
                 cache_dir: Optional[str] = None,
                 maximum_images: Optional[int] = None,
                 shuffle: bool = False,
                 is_ct: bool = False,
                 ct_window: Tuple[float, float] = (-1000, 1000),
                 quantile_range: Tuple[float, float] = (0.0, 0.99),
                 use_cache: bool = True):

        self.read_type = read_type
        self.name = name
        self.data = data
        self.input_shape = input_shape
        self.is_ct = is_ct
        self.ct_window = ct_window
        self.quantile_range = quantile_range
        self.use_cache = use_cache

        # Detect optional data fields from JSON entries
        self.has_segmentation = any('segmentation' in item for item in data)
        self.has_mask = any('mask' in item for item in data)

        # Build per-image modality map
        self._modality_map = {}
        for item in data:
            mod = item.get('modality')
            if mod is not None:
                if mod.lower() not in ('ct', 'mri'):
                    raise ValueError(f"Invalid modality '{mod}' for {item['image']}. Must be 'ct' or 'mri'.")
                self._modality_map[item['image']] = mod.lower() == 'ct'
        self._modality_hash = _deterministic_hash(self._modality_map)

        if self._modality_map and len(self._modality_map) < len(data):
            missing = [item['image'] for item in data if item['image'] not in self._modality_map]
            fallback = "CT" if self.is_ct else "MRI"
            logger.warning(
                f"Dataset '{name}': {len(missing)} of {len(data)} entries have no 'modality' field "
                f"and will fall back to dataset-level is_ct={self.is_ct} ({fallback} preprocessing). "
                f"First missing: {missing[0]}"
            )

        if not data:
            raise ValueError(f"Dataset {name}: 'data' must be provided (from JSON source)")

        if read_type == "itk":
            self.read_image = self.read_image_itk
        elif read_type == "dicom":
            self.read_image = self.read_image_dicom
        else:
            raise ValueError(f"Invalid read_type: {read_type}. Must be 'itk' or 'dicom'")

        # Build field maps for segmentation and mask paths
        self._segmentation_map = {}
        self._mask_map = {}
        if self.has_segmentation:
            self._segmentation_map = {item['image']: item['segmentation']
                                      for item in data if 'segmentation' in item}
        if self.has_mask:
            self._mask_map = {item['image']: item['mask']
                              for item in data if 'mask' in item}

        # Cache setup
        self._cache_path = None
        if use_cache:
            if cache_dir:
                self._cache_path = os.path.join(cache_dir, self.name + "_cached_dataset.trch")
            else:
                self._cache_path = os.path.join(footsteps.output_dir, self.name + "_cached_dataset.trch")

        # Load images
        if self._cache_path and os.path.exists(self._cache_path):
            loaded_cache = torch.load(self._cache_path, map_location="cpu", weights_only=False)
            _validate_cache(loaded_cache, self.name, maximum_images, self.read_type,
                            self.is_ct, self.ct_window, self.quantile_range,
                            self._modality_hash)
            self.store = loaded_cache["store"]
        else:
            self.store = {}
            paths = self.get_image_paths()
            if shuffle:
                random.shuffle(paths)
            if maximum_images:
                paths = paths[:maximum_images]

            failed = 0
            for path in tqdm(paths):
                try:
                    self.store[path] = {"image": self.preprocess_image(path)}
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    failed += 1

            if failed == len(paths):
                raise RuntimeError(f"Dataset {self.name}: all {failed} images failed to load")
            if failed > 0:
                pct = failed / len(paths) * 100
                logger.warning(f"{failed}/{len(paths)} ({pct:.1f}%) images failed to load")

            if self._cache_path:
                os.makedirs(os.path.dirname(os.path.abspath(self._cache_path)), exist_ok=True)
                torch.save(
                    {
                        "name": self.name,
                        "maximum_images": maximum_images,
                        "store": self.store,
                        "read_type": self.read_type,
                        "is_ct": self.is_ct,
                        "ct_window": self.ct_window,
                        "quantile_range": self.quantile_range,
                        "modality_hash": self._modality_hash,
                    },
                    self._cache_path,
                )

        self.keys = list(self.store.keys())
        if len(self.keys) < 2:
            raise ValueError(f"Dataset '{self.name}': need at least 2 images for registration pairs, got {len(self.keys)}")
        logger.info(f"Dataset '{self.name}': {len(self.keys)} images loaded")

        # Load segmentations and masks
        if self.has_segmentation:
            self._load_label_maps("segmentation", self._segmentation_map)
        if self.has_mask:
            self._load_label_maps("mask", self._mask_map)

    def _load_label_maps(self, field_name: str, path_map: Dict[str, str]):
        """Load and cache segmentation or mask label maps for all images."""
        cache_path = None
        if self._cache_path:
            cache_path = self._cache_path.replace("_cached_dataset.trch", f"_cached_{field_name}s.trch")

        if cache_path and os.path.exists(cache_path):
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            cached_shape = cache.get("input_shape")
            cached_data = cache.get("data", {})
            if cached_shape == list(self.input_shape) and all(path in cached_data for path in self.keys):
                for path in self.keys:
                    self.store[path][field_name] = cached_data[path]
                return

        failed_keys = []
        for path in tqdm(self.keys, desc=f"Processing {field_name}s for {self.name}"):
            label_path = path_map.get(path)
            if label_path is None:
                logger.warning(f"No {field_name} for {path}")
                failed_keys.append(path)
                continue
            try:
                self.store[path][field_name] = self._preprocess_label_map(label_path)
            except Exception as e:
                logger.warning(f"Failed to process {field_name} for {path}: {e}")
                failed_keys.append(path)

        for path in failed_keys:
            self.keys.remove(path)
            del self.store[path]
        if failed_keys:
            logger.warning(f"Removed {len(failed_keys)} images with failed {field_name}s")

        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save(
                {
                    "input_shape": list(self.input_shape),
                    "data": {path: self.store[path][field_name] for path in self.keys},
                },
                cache_path,
            )

    def _preprocess_label_map(self, path: str) -> torch.Tensor:
        """Read and resize a label map (segmentation or mask) with nearest interpolation."""
        label = self.read_image_itk(path)
        label = label[None, None].float()
        label = torch.nn.functional.interpolate(label, self.input_shape, mode="nearest")
        return label[0]

    def get_image_paths(self) -> List[str]:
        return [item['image'] for item in self.data]

    def read_image_itk(self, path: str):
        itk_image = reorient(itk.imread(path, itk.F))
        image = itk.GetArrayFromImage(itk_image)
        image = torch.tensor(image)
        return image

    def read_image_dicom(self, path: str):
        namesGenerator = itk.GDCMSeriesFileNames.New()
        namesGenerator.SetUseSeriesDetails(True)
        namesGenerator.SetDirectory(path)
        seriesUID = namesGenerator.GetSeriesUIDs()

        if len(seriesUID) == 0:
            raise ValueError(f"{path}: no DICOM series found in directory")
        if len(seriesUID) > 1:
            logger.warning(f"{path}: {len(seriesUID)} DICOM series found, using first")

        dicom_files = namesGenerator.GetFileNames(seriesUID[0])

        reader = itk.ImageSeriesReader[itk.Image[itk.F, 3]].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(dicom_files)
        reader.Update()
        image = reader.GetOutput()
        image = reorient(image)

        if (
            "ITK_non_uniform_sampling_deviation"
            in image.GetMetaDataDictionary().GetKeys()
        ):
            spacing_deviation = image.GetMetaDataDictionary().Get(
                "ITK_non_uniform_sampling_deviation"
            )
            spacing_deviation = (
                itk.MetaDataObject[itk.D]
                .cast(spacing_deviation)
                .GetMetaDataObjectValue()
            )

            if spacing_deviation > 5:
                raise ValueError(f"{path}: image has non-uniform-spacing: likely a mish-mash")

        image_array = itk.GetArrayFromImage(image)
        image_tensor = torch.tensor(image_array)

        if np.any(np.array(image_array.shape) < 20):
            raise ValueError(f"{path}: image too low resolution")

        return image_tensor

    def preprocess_image(self, path: str):
        image = self.read_image(path)

        image = image[None, None]
        image = image.float()
        image = torch.nn.functional.interpolate(
            image, self.input_shape, mode="trilinear", align_corners=True
        )

        is_ct = self._modality_map.get(path, self.is_ct)
        im_min = self.ct_window[0] if is_ct else torch.quantile(image.view(-1), self.quantile_range[0])
        im_max = self.ct_window[1] if is_ct else torch.quantile(image.view(-1), self.quantile_range[1])

        image = torch.clip(image, im_min, im_max)
        image = image - im_min
        intensity_range = im_max - im_min
        if intensity_range > 0:
            image = image / intensity_range

        return image[0]

    def _pack(self, tensor: torch.Tensor):
        """Compress a tensor for in-memory storage."""
        if _HAS_BLOSC:
            return blosc.pack_array(tensor.numpy())
        return tensor

    def _unpack(self, packed) -> torch.Tensor:
        """Decompress stored data back to a tensor."""
        if _HAS_BLOSC and isinstance(packed, bytes):
            return torch.from_numpy(blosc.unpack_array(packed))
        return packed

    def compress(self):
        """Compress all stored tensors with blosc for memory-efficient storage.
        Enables parallel decompression via DataLoader workers."""
        if not _HAS_BLOSC:
            logger.warning("blosc not available, skipping compression. Install with: pip install blosc")
            return self
        for key in self.store:
            for tensor_key in self.store[key]:
                val = self.store[key][tensor_key]
                if torch.is_tensor(val):
                    self.store[key][tensor_key] = self._pack(val)
        logger.info(f"Dataset '{self.name}': compressed with blosc")
        return self

    def get_image(self, key: str) -> torch.Tensor:
        return self._unpack(self.store[key]["image"])

    def get_key_pair(self) -> Tuple[str, str]:
        return tuple(random.sample(self.keys, 2))

    def get_pair(self) -> Dict[str, torch.Tensor]:
        key_a, key_b = self.get_key_pair()
        result = {
            "image_A": self.get_image(key_a),
            "image_B": self.get_image(key_b),
        }
        if self.has_segmentation:
            result["segmentation_A"] = self._unpack(self.store[key_a]["segmentation"])
            result["segmentation_B"] = self._unpack(self.store[key_b]["segmentation"])
        if self.has_mask:
            result["mask_A"] = self._unpack(self.store[key_a]["mask"])
            result["mask_B"] = self._unpack(self.store[key_b]["mask"])
        return result

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return self.get_pair()


class PairedDataset(Dataset):
    """Paired dataset that returns image pairs from the same subject.

    Accepts all parameters from Dataset. Data entries must include ``subject_id``
    with at least 2 images per subject.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pair_candidates, self.keys = _build_pair_lookup(self.data, self.store, self.keys)
        if not self.keys:
            raise ValueError(
                f"Dataset '{self.name}': no valid pairs found. "
                f"Ensure data entries have 'subject_id' and at least 2 images per subject."
            )
        logger.info(f"Dataset '{self.name}': {len(self.keys)} paired images")

    def get_key_pair(self):
        key1 = random.choice(self.keys)
        return (key1, random.choice(self.pair_candidates[key1]))
