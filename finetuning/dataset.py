import torch
import numpy as np
import json
import collections
from tqdm import tqdm
import random
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
                 data: List[Dict[str, str]],
                 read_type: str = "itk",
                 cache_filename: Optional[str] = None,
                 maximum_images: Optional[int] = None,
                 shuffle: bool = False,
                 is_ct: bool = False,
                 ct_window: Tuple[float, float] = (-1000, 1000),
                 quantile_range: Tuple[float, float] = (0.01, 0.99),
                 use_cache: bool = True):
        
        self.read_type = read_type
        self.name = name
        self.data = data
        self.input_shape = input_shape
        self.is_ct = is_ct
        self.ct_window = ct_window
        self.quantile_range = quantile_range
        self.use_cache = use_cache
        
        if not data:
            raise ValueError(f"Dataset {name}: 'data' must be provided (from JSON source)")
        
        if read_type == "itk":
            self.read_image = self.read_image_itk
        elif read_type == "sitk":
            self.read_image = self.read_image_sitk
        elif read_type == "dicom":
            self.read_image = self.read_image_dicom
        else:
            raise ValueError(f"Invalid read_type: {read_type}. Must be 'itk', 'sitk', or 'dicom'")

        if not use_cache:
            print(f"Loading images without cache...")
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
        elif not cache_filename:
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
            cache_path = cache_filename + "/" + self.name + "_cached_dataset.trch"
            if os.path.exists(cache_path):
                loaded_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
                
                assert self.name == loaded_cache["name"]
                assert maximum_images == loaded_cache["maximum_images"]
                assert self.read_type == loaded_cache["read_type"]
                assert self.is_ct == loaded_cache["is_ct"]
                if self.is_ct:
                    assert self.ct_window == loaded_cache["ct_window"]
                else:
                    assert self.quantile_range == loaded_cache["quantile_range"]
                
                paths = self.get_image_paths()
                self.store = loaded_cache["store"]
            else:
                os.makedirs(cache_filename, exist_ok=True)
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
                        "maximum_images": maximum_images,
                        "store": self.store,
                        "read_type": self.read_type,
                        "is_ct": self.is_ct,
                        "ct_window": self.ct_window,
                        "quantile_range": self.quantile_range,
                    },
                    cache_path,
                )
            
        self.keys = list(self.store.keys())
        print("Image count: ", len(self.keys))

    def get_image_paths(self) -> List[str]:
        return [item['image'] for item in self.data]

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
        namesGenerator = itk.GDCMSeriesFileNames.New()
        namesGenerator.SetUseSeriesDetails(True)
        namesGenerator.SetDirectory(path)
        seriesUID = namesGenerator.GetSeriesUIDs()

        dicom_files = namesGenerator.GetFileNames(seriesUID[0])

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
            image, self.input_shape, mode="trilinear"
        )

        im_min = self.ct_window[0] if self.is_ct else torch.quantile(image.view(-1), self.quantile_range[0])
        im_max = self.ct_window[1] if self.is_ct else torch.quantile(image.view(-1), self.quantile_range[1])
        
        image = torch.clip(image, im_min, im_max)
        image = image - im_min
        image = image / (im_max - im_min)

        return image[0]

    def get_image(self, key: str) -> torch.Tensor:
        unprepped_image = self.store[key]["image"]
        return unprepped_image

    def get_key_pair(self) -> Tuple[str, str]:
        return (random.choice(self.keys), random.choice(self.keys))

    def get_pair(self):
        pair = self.get_key_pair()
        return self.get_image(pair[0]), self.get_image(pair[1])
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        return self.get_pair()


class PairedDataset(Dataset):
    def __init__(
        self,
        input_shape,
        name: str,
        data: List[Dict[str, str]],
        cache_filename=None,
        maximum_images=None,
        is_ct: bool = False,
        ct_window: Tuple[float, float] = (-1000, 1000),
        quantile_range: Tuple[float, float] = (0.01, 0.99),
        read_type: str = "itk",
        shuffle: bool = False,
        use_cache: bool = True,
    ):
        super().__init__(
            input_shape,
            name,
            data,
            cache_filename=cache_filename,
            maximum_images=maximum_images,
            is_ct=is_ct,
            ct_window=ct_window,
            quantile_range=quantile_range,
            read_type=read_type,
            shuffle=shuffle,
            use_cache=use_cache,
        )

        self.pair_lookup = collections.defaultdict(list)
        self.pair_keys = {}

        for item in self.data:
            path = item['image']
            subject_id = item.get('subject_id')
            if path in self.store and subject_id:
                self.pair_keys[path] = subject_id
                self.pair_lookup[subject_id].append(path)
        
        self.keys = [k for k in self.keys if k in self.pair_keys and len(self.pair_lookup[self.pair_keys[k]]) > 1]
        print("Paired image count: ", len(self.keys))

    def get_key_pair(self):
        image_key_1 = random.choice(self.keys)
        subject_id = self.pair_keys[image_key_1]
        candidates = [k for k in self.pair_lookup[subject_id] if k != image_key_1]
        image_key_2 = random.choice(candidates)
        return (image_key_1, image_key_2)

class ImageSegmentationDataset(Dataset):
    def __init__(self,
                 input_shape: Tuple[int, ...],
                 name: str,
                 data: List[Dict[str, str]],
                 read_type: str = "itk",
                 cache_filename: Optional[str] = None,
                 maximum_images: Optional[int] = None,
                 shuffle: bool = False,
                 is_ct: bool = False,
                 ct_window: Tuple[float, float] = (-1000, 1000),
                 quantile_range: Tuple[float, float] = (0.01, 0.99),
                 use_cache: bool = True):
        
        self.segmentation_map = {item['image']: item['segmentation'] for item in data}
        
        super().__init__(input_shape=input_shape,
                         name=name,
                         data=data,
                         read_type=read_type,
                         cache_filename=cache_filename,
                         maximum_images=maximum_images,
                         shuffle=shuffle,
                         is_ct=is_ct,
                         ct_window=ct_window,
                         quantile_range=quantile_range,
                         use_cache=use_cache)
        
        for path in self.keys:
            try:
                self.store[path]["segmentation"] = self.preprocess_segmentation(path)
            except Exception as e:
                print(f"Failed to process segmentation for {path}: {e}")
    
    def get_image_paths(self) -> List[str]:
         return [item['image'] for item in self.data]

    def get_segmentation_path(self, image_path: str) -> Optional[str]:
         return self.segmentation_map.get(image_path)

    def preprocess_segmentation(self, image_path: str) -> torch.Tensor:
        seg_path = self.get_segmentation_path(image_path)
        seg = self.read_image(seg_path)
        seg = seg[None, None].float()
        seg = torch.nn.functional.interpolate(seg, self.input_shape, mode="nearest")
        return seg[0]

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
        
class PairedImageSegmentationDataset(ImageSegmentationDataset):
    def __init__(self,
                 input_shape: Tuple[int, ...],
                 name: str,
                 data: List[Dict[str, str]],
                 read_type: str = "itk",
                 cache_filename: Optional[str] = None,
                 maximum_images: Optional[int] = None,
                 shuffle: bool = False,
                 is_ct: bool = False,
                 ct_window: Tuple[float, float] = (-1000, 1000),
                 quantile_range: Tuple[float, float] = (0.01, 0.99),
                 use_cache: bool = True):
        
        super().__init__(input_shape=input_shape,
                         name=name,
                         data=data,
                         read_type=read_type,
                         cache_filename=cache_filename,
                         maximum_images=maximum_images,
                         shuffle=shuffle,
                         is_ct=is_ct,
                         ct_window=ct_window,
                         quantile_range=quantile_range,
                         use_cache=use_cache)

        self.pair_lookup = collections.defaultdict(list)
        self.pair_keys = {}
        
        for item in self.data:
            path = item['image']
            subject_id = item.get('subject_id')
            if path in self.store and subject_id:
                self.pair_keys[path] = subject_id
                self.pair_lookup[subject_id].append(path)

        self.keys = [k for k in self.keys if k in self.pair_keys and len(self.pair_lookup[self.pair_keys[k]]) > 1]
        print("Paired segmentation count:", len(self.keys))

    def get_key_pair(self) -> Tuple[str, str]:
        image_key_1 = random.choice(self.keys)
        subject_id = self.pair_keys[image_key_1]
        candidates = [k for k in self.pair_lookup[subject_id] if k != image_key_1]
        image_key_2 = random.choice(candidates)
        return (image_key_1, image_key_2)

    def get_pair(self):
        k1, k2 = self.get_key_pair()
        return (
            self.get_image(k1),
            self.get_image(k2),
            self.get_segmentation(k1),
            self.get_segmentation(k2),
        )
