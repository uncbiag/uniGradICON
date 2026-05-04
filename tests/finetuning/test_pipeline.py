"""End-to-end pipeline smoke test: builds DataLoaders from a YAML config and
draws real batches from both train and validation loaders.

This is the "would unigradicon-finetune actually start?" guard. It covers
config loading + validation, schema parsing, dataset construction, cache
plumbing, sampler creation, DataLoader assembly, and ``__getitem__`` /
``_build_pair`` collation in a single pass. ITK image reading is mocked so
the test runs without real medical-image files (~5 seconds).
"""
import json
import pytest
import torch
import yaml

from unigradicon.finetuning import config, dataset
from unigradicon.finetuning.dataset import Fields, PairKeys


def _write_dataset_files(tmp_path, n_images=4, with_segmentation=False):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    entries = []
    for i in range(n_images):
        img = img_dir / f"img_{i:02d}.nii"
        img.write_bytes(b"x")
        entry = {"image": str(img)}
        if with_segmentation:
            seg = img_dir / f"seg_{i:02d}.nii"
            seg.write_bytes(b"x")
            entry["segmentation"] = str(seg)
        entries.append(entry)
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump({"data": entries}, f)
    return json_path


def _write_config(tmp_path, json_path, **training_overrides):
    cfg = {
        "experiment": {"name": "smoke_test", "model_weights": "unigradicon"},
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "input_shape": [8, 8, 8],
            "samples_per_epoch": 4,
            "num_workers": 0,
            **training_overrides,
        },
        "datasets": [{
            "name": "test_ds",
            "type": "unpaired",
            "json_file": str(json_path),
            "cache_dir": str(tmp_path / "cache"),
        }],
    }
    yaml_path = tmp_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return yaml_path


def test_pipeline_rejects_zero_iterations_per_epoch(tmp_path, fake_image_reader):
    """``samples_per_epoch < batch_size * num_gpus`` would yield zero
    training iterations per epoch; the loader build must surface this
    rather than let training silently no-op."""
    json_path = _write_dataset_files(tmp_path, n_images=4)
    yaml_path = _write_config(tmp_path, json_path, batch_size=8, samples_per_epoch=4)

    with pytest.raises(ValueError, match="zero training iterations"):
        config.create_data_loaders(str(yaml_path))


def test_pipeline_starts_minimal_config(tmp_path, fake_image_reader):
    """Smallest possible valid config: images only, single dataset, 1 epoch."""
    json_path = _write_dataset_files(tmp_path)
    yaml_path = _write_config(tmp_path, json_path)

    bundle = config.create_data_loaders(str(yaml_path))

    assert bundle.train_loader is not None
    assert "test_ds" in bundle.val_loaders
    assert bundle.data_fields == frozenset()


def test_pipeline_train_batch_shape(tmp_path, fake_image_reader):
    """The train loader must yield batches with image_A / image_B at the
    configured input_shape."""
    json_path = _write_dataset_files(tmp_path)
    yaml_path = _write_config(tmp_path, json_path)

    bundle = config.create_data_loaders(str(yaml_path))
    batch = next(iter(bundle.train_loader))

    assert PairKeys.IMAGE_A in batch
    assert PairKeys.IMAGE_B in batch
    image_a = batch[PairKeys.IMAGE_A]
    image_b = batch[PairKeys.IMAGE_B]
    assert image_a.shape[0] == 2, f"expected batch_size=2, got {image_a.shape}"
    assert image_a.shape[-3:] == (8, 8, 8)
    assert image_a.shape == image_b.shape


def test_pipeline_val_batch_shape(tmp_path, fake_image_reader):
    """The validation loader uses batch_size=1 by default."""
    json_path = _write_dataset_files(tmp_path)
    yaml_path = _write_config(tmp_path, json_path)

    bundle = config.create_data_loaders(str(yaml_path))
    val_loader = bundle.val_loaders["test_ds"]
    batch = next(iter(val_loader))

    assert batch[PairKeys.IMAGE_A].shape[0] == 1
    assert batch[PairKeys.IMAGE_A].shape[-3:] == (8, 8, 8)


def test_pipeline_with_segmentation_includes_seg_in_batch(tmp_path, fake_image_reader):
    """When dice_loss_weight > 0, batches include segmentation tensors."""
    json_path = _write_dataset_files(tmp_path, with_segmentation=True)
    yaml_path = _write_config(tmp_path, json_path, dice_loss_weight=0.5)

    bundle = config.create_data_loaders(str(yaml_path))
    assert Fields.SEGMENTATION in bundle.data_fields

    batch = next(iter(bundle.train_loader))
    assert PairKeys.SEGMENTATION_A in batch
    assert PairKeys.SEGMENTATION_B in batch
    assert batch[PairKeys.SEGMENTATION_A].shape == batch[PairKeys.IMAGE_A].shape


def test_pipeline_iterates_full_epoch(tmp_path, fake_image_reader):
    """The train loader iterates exactly samples_per_epoch / effective_batch
    times, covering a full epoch without exception."""
    json_path = _write_dataset_files(tmp_path)
    yaml_path = _write_config(tmp_path, json_path)

    bundle = config.create_data_loaders(str(yaml_path))
    iterations = sum(1 for _ in bundle.train_loader)
    # samples_per_epoch=4, batch_size=2, single GPU yields 2 iterations.
    assert iterations == 2


def test_pipeline_seed_yields_reproducible_first_batch(tmp_path, fake_image_reader):
    """With seed set and num_workers=0, two independent runs must produce
    the same first-batch anchor selection. The CLI seeds before
    ``create_data_loaders``; the test mirrors that flow explicitly."""
    json_path = _write_dataset_files(tmp_path, n_images=10)
    yaml_path = _write_config(tmp_path, json_path, seed=42)

    config.set_reproducibility_seed(42)
    bundle_a = config.create_data_loaders(str(yaml_path))
    indices_a = list(iter(bundle_a.train_loader.sampler))

    config.set_reproducibility_seed(42)
    bundle_b = config.create_data_loaders(str(yaml_path))
    indices_b = list(iter(bundle_b.train_loader.sampler))

    assert indices_a == indices_b, (
        "WeightedRandomSampler index sequence is not reproducible under "
        "set_reproducibility_seed; check torch.manual_seed wiring."
    )
