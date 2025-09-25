import torch
import numpy as np
import re as regex
import collections
from tqdm import tqdm
import random
import glob
import os
import footsteps
import itk
import SimpleITK
from typing import List, Tuple, Optional, Union, Dict, Any


def reorient(moving):
    desired_coordinate_orientation = itk.ITKCommonBasePython.itkSpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_RAS

    if hasattr(itk, "AnatomicalOrientation"):
        desired_coordinate_orientation = itk.AnatomicalOrientation(desired_coordinate_orientation)

    return itk.orient_image_filter(
        moving, 
        desired_coordinate_orientation=desired_coordinate_orientation,
        use_image_direction=True)

class Dataset:
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 name: str,
                 image_glob: str,
                 read_type: str = "itk",
                 cache_filename: Optional[str] = None,
                 maximum_images: Optional[int] = None,
                 shuffle: bool = False,
                 is_ct: bool = False,
                 ct_window: Tuple[float, float] = (-1000, 1000),
                 quantile_range: Tuple[float, float] = (0.01, 0.99)):
        
        self.read_type = read_type
        self.name = name
        self.image_glob = image_glob
        self.input_shape = input_shape
        self.is_ct = is_ct
        self.ct_window = ct_window
        self.quantile_range = quantile_range
        
        # Set the appropriate read method based on read_type
        if read_type == "itk":
            self.read_image = self.read_image_itk
        elif read_type == "sitk":
            self.read_image = self.read_image_sitk
        elif read_type == "dicom":
            self.read_image = self.read_image_dicom
        else:
            raise ValueError(f"Invalid read_type: {read_type}. Must be 'itk', 'sitk', or 'dicom'")

        if not cache_filename:
            self.store = {}
            paths = self.get_image_paths()
            if shuffle:
                random.shuffle(paths)
            if maximum_images:
                paths = paths[:maximum_images]
            for path in tqdm(paths):
                try:
                    self.store[path] = {"image": self.preprocess_image(path)}
                except Exception as e:
                    print(e)

            torch.save(
                {
                    "name": self.name,
                    "image_glob": self.image_glob,
                    "maximum_images": maximum_images,
                    "store": self.store,
                    "read_type": self.read_type,
                    "is_ct": self.is_ct,
                    "ct_window": self.ct_window,
                    "quantile_range": self.quantile_range,
                },
                footsteps.output_dir + self.name + "_cached_dataset.trch",
            )
        else:
            loaded_cache = torch.load(cache_filename + "/" + self.name + "_cached_dataset.trch", map_location="cpu")
            
            assert self.name == loaded_cache["name"]
            assert maximum_images == loaded_cache["maximum_images"]
            assert self.image_glob == loaded_cache["image_glob"]
            assert self.read_type == loaded_cache["read_type"]
            assert self.is_ct == loaded_cache["is_ct"]
            if self.is_ct:
                assert self.ct_window == loaded_cache["ct_window"]
            else:
                assert self.quantile_range == loaded_cache["quantile_range"]
            
            paths = self.get_image_paths()
            self.store = loaded_cache["store"]
            
        self.keys = list(self.store.keys())
        print("Image count: ", len(self.keys))

    def get_image_paths(self) -> List[str]:
        return list(glob.glob(self.image_glob))

    def read_image_sitk(self, path: str):
        itk_image = SimpleITK.ReadImage(path)
        image = SimpleITK.GetArrayFromImage(itk_image)
        image = torch.tensor(image)
        return image[0]

    def read_image_itk(self, path: str):
        itk_image = reorient(itk.imread(path))
        image = itk.GetArrayFromImage(itk_image)
        image = torch.tensor(image)
        return image
    
    def read_image_dicom(self, path: str):
        """
        Reads a DICOM series from a directory path and returns it as a tensor.
        
        Args:
            path (str): Directory containing DICOM files
                e.g., "files/image342/"
        
        Returns:
            torch.Tensor: 3D tensor containing the DICOM volume
        """
        namesGenerator = itk.GDCMSeriesFileNames.New()
        namesGenerator.SetUseSeriesDetails(True)
        namesGenerator.SetDirectory(path)
        seriesUID = namesGenerator.GetSeriesUIDs()

        dicom_files = namesGenerator.GetFileNames(seriesUID[0])

        # Read the DICOM series as a 3D image
        reader = itk.ImageSeriesReader[itk.Image[itk.SS, 3]].New()
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

        # Convert to tensor
        image_array = itk.GetArrayFromImage(image)
        image_tensor = torch.tensor(image_array)
    
        if np.any(np.array(image_array.shape) < 20):
            raise ValueError(f"{path}: image too low resolution")

        return image_tensor
    
    def preprocess_image(self, path: str):
        """
        Load an image, crop away any black bars, and then resize to the target resolution.
        """
        image = self.read_image(path)

        image = image[None, None]
        image = image.float()
        image = torch.nn.functional.interpolate(
            image, self.input_shape[2:], mode="trilinear"
        )

        im_min = self.ct_window[0] if self.is_ct else torch.quantile(image.view(-1), self.quantile_range[0])
        im_max = self.ct_window[1] if self.is_ct else torch.quantile(image.view(-1), self.quantile_range[1])
        
        image = torch.clip(image, im_min, im_max)
        image = image - im_min
        image = image / (im_max - im_min)

        return image

    def get_image(self, key: str) -> torch.Tensor:
        unprepped_image = self.store[key]["image"]
        return unprepped_image

    def get_key_pair(self) -> Tuple[str, str]:
        """
        get a pair of images from the dataset.
        This is the one that should be overridden to differentiate between
        paired and unpaired datasets.
        """
        return (random.choice(self.keys), random.choice(self.keys))

    def get_pair(self):
        pair = self.get_key_pair()
        return self.get_image(pair[0]), self.get_image(pair[1])


class PairedDataset(Dataset):
    def __init__(
        self,
        input_shape,
        name: str,
        image_glob: str,
        cache_filename=None,
        maximum_images=None,
        match_regex=None,
        is_ct: bool = False,
        ct_window: Tuple[float, float] = (-1000, 1000),
        quantile_range: Tuple[float, float] = (0.01, 0.99),
        read_type: str = "itk",
        shuffle: bool = False,      
    ):
        super().__init__(
            input_shape,
            name,
            image_glob,
            cache_filename=cache_filename,
            maximum_images=maximum_images,
            is_ct=is_ct,
            ct_window=ct_window,
            quantile_range=quantile_range,
            read_type=read_type,
            shuffle=shuffle,
        )
        if match_regex == None:
            raise NotImplementedError()

        self.pair_lookup = collections.defaultdict(lambda: [])
        self.pair_keys = {}

        for key in self.store.keys():
            pair_key = regex.search(match_regex, key).group(1)
            self.pair_keys[key] = pair_key
            self.pair_lookup[pair_key].append(key)
            
        #keep only keys that have at least one pair
        self.keys = [k for k in self.keys if len(self.pair_lookup[self.pair_keys[k]]) > 1]
        print("Paired image count: ", len(self.keys))

    def get_key_pair(self):
        image_key_1 = random.choice(self.keys)
        candidates = [k for k in self.pair_lookup[self.pair_keys[image_key_1]] if k != image_key_1]
        image_key_2 = random.choice(candidates)
        return (image_key_1, image_key_2)

class SegmentationDataset(Dataset):
    def __init__(self,
                 input_shape: Tuple[int, ...],
                 name: str,
                 image_glob: str,
                 segmentation_glob: str,
                 read_type: str = "itk",
                 cache_filename: Optional[str] = None,
                 maximum_images: Optional[int] = None,
                 shuffle: bool = False,
                 is_ct: bool = False,
                 ct_window: Tuple[float, float] = (-1000, 1000),
                 quantile_range: Tuple[float, float] = (0.01, 0.99)):
        # Call parent constructor
        super().__init__(input_shape=input_shape,
                         name=name,
                         image_glob=image_glob,
                         read_type=read_type,
                         cache_filename=cache_filename,
                         maximum_images=maximum_images,
                         shuffle=shuffle,
                         is_ct=is_ct,
                         ct_window=ct_window,
                         quantile_range=quantile_range)

        self.segmentation_glob = segmentation_glob

        # Precompute segmentation lookup: basename -> full path
        self.segmentation_map: Dict[str, str] = {
            os.path.basename(p): p for p in glob.glob(self.segmentation_glob)
        }
        
        # Filter and rebuild store so only paired items remain
        for path in self.keys:
            seg_path = self.get_segmentation_path(path)
            if seg_path:  # keep only if segmentation exists
                try:
                    self.store[path]["segmentation"] = self.preprocess_segmentation(path)
                except Exception as e:
                    print(f"Failed to process {path}: {e}")
            else:
                print(f"Skipping {path}, no segmentation found.")
        
        #find keys that have segmentation
        self.store = {k: v for k, v in self.store.items() if "segmentation" in v}
        self.keys = list(self.store.keys())
        print("Paired image count:", len(self.keys))

    def get_segmentation_path(self, image_path: str) -> Optional[str]:
        """Find matching segmentation path via precomputed lookup."""
        return self.segmentation_map.get(os.path.basename(image_path), None)

    def preprocess_segmentation(self, image_path: str) -> torch.Tensor:
        seg_path = self.get_segmentation_path(image_path)
        seg = self.read_image(seg_path)
        seg = seg[None, None].float()
        seg = torch.nn.functional.interpolate(seg, self.input_shape[2:], mode="nearest")
        return seg

    def get_image(self, key: str) -> torch.Tensor:
        return self.store[key]["image"]

    def get_segmentation(self, key: str) -> torch.Tensor:
        return self.store[key]["segmentation"]

    def get_pair(self):
        pair = self.get_key_pair()
        return (
            self.get_image(pair[0]),
            self.get_image(pair[1]),
            self.get_segmentation(pair[0]),
            self.get_segmentation(pair[1]),
        )
        
class PairedSegmentationDataset(SegmentationDataset):
    def __init__(self,
                 input_shape: Tuple[int, ...],
                 name: str,
                 image_glob: str,
                 segmentation_glob: str,
                 match_regex: str,
                 read_type: str = "itk",
                 cache_filename: Optional[str] = None,
                 maximum_images: Optional[int] = None,
                 shuffle: bool = False,
                 is_ct: bool = False,
                 ct_window: Tuple[float, float] = (-1000, 1000),
                 quantile_range: Tuple[float, float] = (0.01, 0.99)):
        # Initialize SegmentationDataset first
        super().__init__(input_shape=input_shape,
                         name=name,
                         image_glob=image_glob,
                         segmentation_glob=segmentation_glob,
                         read_type=read_type,
                         cache_filename=cache_filename,
                         maximum_images=maximum_images,
                         shuffle=shuffle,
                         is_ct=is_ct,
                         ct_window=ct_window,
                         quantile_range=quantile_range)

        if match_regex is None:
            raise ValueError("match_regex must be provided for PairedSegmentationDataset")

        # Build pair lookup using regex
        self.pair_lookup = collections.defaultdict(list)
        self.pair_keys = {}
        
        for key in self.store.keys():
            match = regex.search(match_regex, key)
            if match:
                pair_key = match.group(1)
                self.pair_keys[key] = pair_key
                self.pair_lookup[pair_key].append(key)

        # Keep only those keys that actually have a mate
        self.keys = [k for k in self.keys if len(self.pair_lookup[self.pair_keys[k]]) > 1]
        print("Paired segmentation count:", len(self.keys))

    def get_key_pair(self) -> Tuple[str, str]:
        """Get a pair of image/segmentation keys that belong together."""
        image_key_1 = random.choice(self.keys)
        candidates = [k for k in self.pair_lookup[self.pair_keys[image_key_1]] if k != image_key_1]
        image_key_2 = random.choice(candidates)
        return (image_key_1, image_key_2)

    def get_pair(self):
        """Return paired images and their segmentations."""
        k1, k2 = self.get_key_pair()
        return (
            self.get_image(k1),
            self.get_image(k2),
            self.get_segmentation(k1),
            self.get_segmentation(k2),
        )