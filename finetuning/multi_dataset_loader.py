import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader
from typing import Dict, List, Tuple, Any
import config_loader


def create_multi_dataset_loaders(config_path: str) -> Tuple[DataLoader, Dict[str, DataLoader], Dict[str, Any], str]:
    """
    Create training and validation dataloaders from YAML config.
    
    Returns:
        train_loader: DataLoader for training with weighted sampling
        val_loaders: Dict mapping dataset name to its validation DataLoader
        config: The loaded configuration dictionary
        mode: 'standard' or 'segmentation' indicating dataset type
    """
    config = config_loader.load_config(config_path)
    
    mode = config_loader.validate_dataset_consistency(config['datasets'])
    print(f"Dataset mode: {mode}")
    
    input_shape = config['training']['input_shape']
    batch_size = config['training']['batch_size']
    gpus = config['training']['gpus']
    num_gpus = len(gpus)
    
    datasets = []
    weights = []
    val_loaders = {}
    
    print(f"\nLoading {len(config['datasets'])} dataset(s)...")
    for ds_config in config['datasets']:
        print(f"\nProcessing dataset: {ds_config['name']}")
        print(f"  Type: {ds_config['type']}")
        print(f"  Weight: {ds_config.get('weight', 1.0)}")
        
        ds = config_loader.create_dataset_from_config(ds_config, input_shape)
        datasets.append(ds)
        
        # Get dataset length - handle both __len__ and keys attribute
        if hasattr(ds, '__len__'):
            ds_length = len(ds)
        elif hasattr(ds, 'keys'):
            # keys might be empty list initially, but attribute exists
            ds_length = len(ds.keys) if ds.keys else 0
        else:
            raise ValueError(f"Dataset {ds_config['name']} has no way to determine length (no __len__ or keys attribute)")
        
        ds_weight = ds_config.get('weight', 1.0)
        weights.extend([ds_weight] * ds_length)
        
        print(f"  Loaded {ds_length} samples")
        
        val_loaders[ds_config['name']] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )
    
    combined_dataset = ConcatDataset(datasets)
    total_samples = len(combined_dataset)
    print(f"\nTotal combined samples: {total_samples}")
    
    samples_per_epoch = config['training'].get('samples_per_epoch', total_samples)
    
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    print(f"Samples per epoch: {samples_per_epoch}")
    print(f"Creating WeightedRandomSampler")
    print(f"Batch size: {batch_size} x {num_gpus} GPUs = {batch_size * num_gpus} per batch")
    
    train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size * num_gpus,
        num_workers=4,
        drop_last=True,
        sampler=WeightedRandomSampler(
            weights=normalized_weights,
            num_samples=samples_per_epoch,
            replacement=True
        )
    )
    
    effective_batch_size = batch_size * num_gpus
    iterations_per_epoch = samples_per_epoch // effective_batch_size
    
    print(f"\nDataLoaders created successfully!")
    print(f"Training iterations per epoch: {iterations_per_epoch}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation datasets: {list(val_loaders.keys())}")
    
    return train_loader, val_loaders, config, mode


def get_dataset_info(config_path: str) -> Dict[str, Any]:
    """Load config and return summary information about datasets."""
    config = config_loader.load_config(config_path)
    
    info = {
        'experiment_name': config['experiment']['name'],
        'mode': config_loader.validate_dataset_consistency(config['datasets']),
        'num_datasets': len(config['datasets']),
        'datasets': []
    }
    
    for ds_config in config['datasets']:
        info['datasets'].append({
            'name': ds_config['name'],
            'type': ds_config['type'],
            'weight': ds_config.get('weight', 1.0),
            'json_file': ds_config['json_file']
        })
    
    return info

