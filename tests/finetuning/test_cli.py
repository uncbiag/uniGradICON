"""CLI-level tests for the ``unigradicon-finetune`` entry point.

The ``unigradicon-finetune`` console script dispatches to
``unigradicon.finetuning.finetune.main(argv)``. These tests call ``main()``
directly with synthetic argv to exercise the argparse layer and the side
effects that happen before ``finetune_multi`` (config load, validation,
seed setup, footsteps initialization, data-loader build). ``finetune_multi``
itself is mocked so tests do not require the real network or a GPU.
"""
import json

import pytest
import torch
import yaml

from unigradicon.finetuning import dataset, finetune


@pytest.fixture
def isolate_cwd(tmp_path, monkeypatch):
    """Run inside a temp dir so footsteps' results/ dir doesn't pollute the
    project tree, and reset footsteps' module-level singleton flag so multiple
    main() calls in the same session don't trip the "can only be initialized
    once" guard."""
    monkeypatch.chdir(tmp_path)
    import footsteps
    monkeypatch.setattr(footsteps, "initialized", False, raising=False)
    monkeypatch.setattr(footsteps, "output_dir_impl", None, raising=False)
    return tmp_path


@pytest.fixture
def mock_finetune_multi(monkeypatch):
    """Capture the (config, data_loader, val_loaders, data_fields) call so the
    test can assert main() reached the training step. The real
    ``finetune_multi`` requires the unigradicon network, a GPU, and a
    model-zoo download, none of which belong in a unit test."""
    calls = []

    def _capture(*, config, data_loader, val_data_loaders_dict, data_fields):
        calls.append({
            "config": config,
            "data_loader": data_loader,
            "val_loaders": val_data_loaders_dict,
            "data_fields": data_fields,
        })

    monkeypatch.setattr(finetune, "finetune_multi", _capture)
    return calls


def _write_pipeline_files(tmp_path, n_images=4, with_seed=None, **training_overrides):
    """Drop a JSON dataset + YAML config under tmp_path and return the YAML path."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    entries = []
    for i in range(n_images):
        p = img_dir / f"img_{i:02d}.nii"
        p.write_bytes(b"x")
        entries.append({"image": str(p)})
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump({"data": entries}, f)

    training = {
        "batch_size": 2,
        "epochs": 1,
        "input_shape": [8, 8, 8],
        "samples_per_epoch": 4,
        "num_workers": 0,
        **training_overrides,
    }
    if with_seed is not None:
        training["seed"] = with_seed

    cfg = {
        "experiment": {"name": "cli_test", "model_weights": "unigradicon"},
        "training": training,
        "datasets": [{
            "name": "cli_ds",
            "type": "unpaired",
            "json_file": str(json_path),
            "cache_dir": str(tmp_path / "cache"),
        }],
    }
    yaml_path = tmp_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return yaml_path


def test_cli_requires_config_arg():
    """``unigradicon-finetune`` (no args) must fail with a non-zero exit."""
    with pytest.raises(SystemExit) as exc_info:
        finetune.main([])
    assert exc_info.value.code != 0


def test_cli_rejects_unknown_argument():
    with pytest.raises(SystemExit):
        finetune.main(["--config", "x.yaml", "--unknown-flag", "v"])


def test_cli_rejects_missing_config_file(tmp_path, isolate_cwd):
    missing = str(tmp_path / "does_not_exist.yaml")
    with pytest.raises(FileNotFoundError):
        finetune.main(["--config", missing])


def test_cli_starts_finetuning_with_valid_config(
    tmp_path, isolate_cwd, fake_image_reader, mock_finetune_multi
):
    """Happy path: a valid YAML drives main() through to ``finetune_multi``
    with a populated DataLoader bundle."""
    yaml_path = _write_pipeline_files(tmp_path)

    finetune.main(["--config", str(yaml_path)])

    assert len(mock_finetune_multi) == 1
    call = mock_finetune_multi[0]
    assert call["data_loader"] is not None
    assert "cli_ds" in call["val_loaders"]
    assert call["data_fields"] == frozenset()


def test_cli_propagates_invalid_config_errors(
    tmp_path, isolate_cwd, fake_image_reader, mock_finetune_multi
):
    """Schema-validation failures must surface as ValueError before training
    starts (we never reach finetune_multi)."""
    yaml_path = _write_pipeline_files(tmp_path, similarity="not_a_real_one")

    with pytest.raises(ValueError, match="similarity"):
        finetune.main(["--config", str(yaml_path)])
    assert mock_finetune_multi == []


def test_cli_propagates_paired_without_subject_id_error(
    tmp_path, isolate_cwd, fake_image_reader, mock_finetune_multi
):
    """The fast-fail validator catches paired-without-subject_id before any
    image preprocessing happens."""
    yaml_path = _write_pipeline_files(tmp_path)
    # Mutate the YAML to declare 'paired' without subject_id support.
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg["datasets"][0]["type"] = "paired"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises(ValueError, match="paired"):
        finetune.main(["--config", str(yaml_path)])
    assert mock_finetune_multi == []


def test_cli_seed_is_applied_before_data_loader_build(
    tmp_path, isolate_cwd, fake_image_reader, mock_finetune_multi
):
    """When 'seed' is in the YAML, set_reproducibility_seed must run before
    create_data_loaders so any RNG-dependent shuffle is deterministic."""
    import footsteps
    yaml_path = _write_pipeline_files(tmp_path, with_seed=42)
    finetune.main(["--config", str(yaml_path)])

    # Two independent main() invocations under the same seed must produce
    # the same sampler index sequence. Reset footsteps' singleton flag so
    # the second main() call can re-initialize.
    sampler_a = mock_finetune_multi[0]["data_loader"].sampler
    indices_a = list(iter(sampler_a))

    mock_finetune_multi.clear()
    footsteps.initialized = False
    footsteps.output_dir_impl = None
    finetune.main(["--config", str(yaml_path)])
    sampler_b = mock_finetune_multi[0]["data_loader"].sampler
    indices_b = list(iter(sampler_b))

    assert indices_a == indices_b
