import yaml
from typing import Dict, List, Tuple, Any
import dataset


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_dataset_consistency(configs: List[Dict[str, Any]]) -> str:
    """
    Validate that all datasets are consistent (either all with segmentation or all without).
    Returns: 'standard' for datasets without segmentation, 'segmentation' for datasets with segmentation
    Raises: ValueError if datasets are mixed
    """
    seg_types = {'unpaired_with_seg', 'paired_with_seg'}
    standard_types = {'unpaired', 'paired'}
    
    dataset_types = [ds['type'] for ds in configs]
    
    has_seg = any(dt in seg_types for dt in dataset_types)
    has_standard = any(dt in standard_types for dt in dataset_types)
    
    if has_seg and has_standard:
        raise ValueError(
            "Cannot mix dataset types with and without segmentations in the same config. "
            f"Found types: {dataset_types}. "
            "Use either all standard types (unpaired, paired) or all segmentation types "
            "(unpaired_with_seg, paired_with_seg)."
        )
    
    if has_seg:
        return 'segmentation'
    else:
        return 'standard'


def create_dataset_from_config(dataset_config: Dict[str, Any], input_shape: Tuple[int, ...]) -> dataset.Dataset:
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
        'image_glob': dataset_config['image_glob'],
        'read_type': dataset_config.get('read_type', 'itk'),
        'cache_filename': dataset_config.get('cache_filename'),
        'maximum_images': dataset_config.get('maximum_images'),
        'shuffle': dataset_config.get('shuffle', True),
        'is_ct': dataset_config.get('is_ct', False),
        'use_cache': dataset_config.get('use_cache', True),
    }
    
    if dataset_config.get('is_ct'):
        common_params['ct_window'] = tuple(dataset_config.get('ct_window', [-1000, 1000]))
    else:
        common_params['quantile_range'] = tuple(dataset_config.get('quantile_range', [0.01, 0.99]))
    
    if dataset_type == 'unpaired':
        return dataset.Dataset(**common_params)
    
    elif dataset_type == 'paired':
        if 'match_regex' not in dataset_config:
            raise ValueError(f"Dataset '{dataset_config['name']}' of type 'paired' requires 'match_regex' parameter")
        return dataset.PairedDataset(
            **common_params,
            match_regex=dataset_config['match_regex']
        )
    
    elif dataset_type == 'unpaired_with_seg':
        if 'segmentation_glob' not in dataset_config:
            raise ValueError(f"Dataset '{dataset_config['name']}' of type 'unpaired_with_seg' requires 'segmentation_glob' parameter")
        return dataset.ImageSegmentationDataset(
            **common_params,
            segmentation_glob=dataset_config['segmentation_glob'],
            seg_match_regex=dataset_config.get('seg_match_regex', None)
        )
    
    elif dataset_type == 'paired_with_seg':
        if 'segmentation_glob' not in dataset_config:
            raise ValueError(f"Dataset '{dataset_config['name']}' of type 'paired_with_seg' requires 'segmentation_glob' parameter")
        if 'match_regex' not in dataset_config:
            raise ValueError(f"Dataset '{dataset_config['name']}' of type 'paired_with_seg' requires 'match_regex' parameter")
        return dataset.PairedImageSegmentationDataset(
            **common_params,
            segmentation_glob=dataset_config['segmentation_glob'],
            match_regex=dataset_config['match_regex'],
            seg_match_regex=dataset_config.get('seg_match_regex', None)
        )
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Must be one of: unpaired, paired, unpaired_with_seg, paired_with_seg")

