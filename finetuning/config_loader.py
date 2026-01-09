import yaml
import json
import os
from typing import Dict, List, Tuple, Any
import dataset


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_json_dataset_file(json_path: str) -> List[Dict[str, str]]:
    """Load dataset definition from JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON dataset file not found: {json_path}")
        
    with open(json_path, 'r') as f:
        content = json.load(f)
    
    if "data" not in content:
        raise ValueError(f"JSON dataset file {json_path} must contain top-level 'data' key.")
        
    return content["data"]


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
    
    # Load JSON data
    if 'json_file' not in dataset_config:
        raise ValueError(f"Dataset '{dataset_config['name']}' must specify 'json_file'")
        
    common_params['data'] = load_json_dataset_file(dataset_config['json_file'])
    
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
