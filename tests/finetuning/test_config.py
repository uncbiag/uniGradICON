"""Unit tests for unigradicon.finetuning.config validators + schema parsing."""
import json
import pytest

from unigradicon.finetuning import config


def test_training_config_lambda_alias():
    """The YAML key 'lambda' must map to the dataclass field 'lmbda'."""
    tc = config.TrainingConfig.from_dict({"lambda": 2.5})
    assert tc.lmbda == 2.5


def test_training_config_unknown_keys_silently_dropped_in_from_dict():
    """from_dict drops unknown keys; ConfigValidator handles the warning."""
    tc = config.TrainingConfig.from_dict({"unknown_key": 1, "lambda": 1.0})
    assert tc.lmbda == 1.0
    assert not hasattr(tc, "unknown_key")


def test_training_config_defaults_when_empty_dict():
    tc = config.TrainingConfig.from_dict({})
    assert tc.batch_size == 4
    assert tc.epochs == 500
    assert tc.lmbda == 1.5


def test_training_config_network_input_shape_property():
    tc = config.TrainingConfig.from_dict({"input_shape": [128, 128, 128]})
    assert tc.network_input_shape == [1, 1, 128, 128, 128]


def test_dataset_config_post_init_rejects_empty_required_fields():
    with pytest.raises(ValueError, match="non-empty 'name'"):
        config.DatasetConfig()
    with pytest.raises(ValueError, match="non-empty 'type'"):
        config.DatasetConfig(name="d")
    with pytest.raises(ValueError, match="non-empty 'json_file'"):
        config.DatasetConfig(name="d", type="unpaired")


def test_dataset_config_from_dict_coerces_lists_to_tuples():
    dc = config.DatasetConfig.from_dict({
        "name": "d", "type": "unpaired", "json_file": "x.json",
        "ct_window": [-500, 500],
        "quantile_range": [0.1, 0.9],
    })
    assert dc.ct_window == (-500, 500)
    assert dc.quantile_range == (0.1, 0.9)
    assert isinstance(dc.ct_window, tuple)
    assert isinstance(dc.quantile_range, tuple)


def _minimal_config(**training_overrides):
    cfg = {
        "experiment": {"name": "demo", "model_weights": "unigradicon"},
        "datasets": [{"name": "d1", "type": "unpaired", "json_file": "x.json"}],
    }
    if training_overrides:
        cfg["training"] = training_overrides
    return cfg


def test_validate_rejects_missing_experiment():
    with pytest.raises(ValueError, match="experiment"):
        config.validate_config({"datasets": [{"name": "d", "type": "unpaired", "json_file": "x"}]})


def test_validate_rejects_missing_datasets():
    with pytest.raises(ValueError, match="datasets"):
        config.validate_config({"experiment": {"name": "d", "model_weights": "unigradicon"}})


def test_validate_rejects_empty_datasets_list():
    with pytest.raises(ValueError, match="datasets"):
        config.validate_config({
            "experiment": {"name": "d", "model_weights": "unigradicon"},
            "datasets": [],
        })


def test_validate_rejects_invalid_similarity():
    with pytest.raises(ValueError, match="similarity"):
        config.validate_config(_minimal_config(similarity="not_a_real_one"))


@pytest.mark.parametrize("sim", ["lncc", "lncc2", "mind", "LNCC", "LNCC2", "MIND"])
def test_validate_accepts_valid_similarities_case_insensitive(sim):
    config.validate_config(_minimal_config(similarity=sim))


def test_validate_rejects_negative_learning_rate():
    with pytest.raises(ValueError, match="learning_rate"):
        config.validate_config(_minimal_config(learning_rate=-1.0))


@pytest.mark.parametrize("key", [
    "eval_period", "save_period", "epochs", "batch_size",
    "lncc_sigma", "mind_radius", "mind_dilation",
])
def test_validate_rejects_non_positive_int_keys(key):
    with pytest.raises(ValueError, match=key):
        config.validate_config(_minimal_config(**{key: 0}))
    with pytest.raises(ValueError, match=key):
        config.validate_config(_minimal_config(**{key: -1}))


@pytest.mark.parametrize("key", ["lambda", "dice_loss_weight", "num_workers"])
def test_validate_rejects_negative_non_negative_keys(key):
    with pytest.raises(ValueError, match=key):
        config.validate_config(_minimal_config(**{key: -1}))


@pytest.mark.parametrize("shape", [
    [0, 128, 128],
    [128, -1, 128],
    [128, 128],
    [128, 128, 128, 128],
    "not_a_list",
    [128, 128, "x"],
])
def test_validate_rejects_bad_input_shape(shape):
    with pytest.raises(ValueError, match="input_shape"):
        config.validate_config(_minimal_config(input_shape=shape))


@pytest.mark.parametrize("gpus", [
    [],
    "0",
    ["0", "1"],
    [0, -1],
    [0, 1.0],
])
def test_validate_rejects_bad_gpus(gpus):
    with pytest.raises(ValueError, match="gpus"):
        config.validate_config(_minimal_config(gpus=gpus))


def test_validate_accepts_valid_gpus():
    config.validate_config(_minimal_config(gpus=[0, 1, 2]))


@pytest.mark.parametrize("spe", [0, -1, "100", 1.5])
def test_validate_rejects_bad_samples_per_epoch(spe):
    with pytest.raises(ValueError, match="samples_per_epoch"):
        config.validate_config(_minimal_config(samples_per_epoch=spe))


def test_validate_accepts_null_or_positive_samples_per_epoch():
    config.validate_config(_minimal_config(samples_per_epoch=None))
    config.validate_config(_minimal_config(samples_per_epoch=128))


def test_paired_check_message_when_no_subject_id_field(tmp_path):
    """When no entry has a subject_id at all, the error message names the
    missing field directly."""
    json_path = _write_json(tmp_path, "data.json", [
        {"image": "a.nii"}, {"image": "b.nii"},
    ])
    schema = _schema_with_paired_dataset(tmp_path, json_path)
    cache = config.DatasetJsonCache()
    with pytest.raises(ValueError, match="subject_id"):
        config.validate_paired_datasets_have_pairs(schema, str(tmp_path), cache)


def test_load_entries_tolerates_null_optional_field(tmp_path):
    """``{"segmentation": null}`` should be treated as absent, not crash."""
    json_path = _write_json(tmp_path, "data.json", [
        {"image": str(tmp_path / "a.nii"), "segmentation": None},
        {"image": str(tmp_path / "b.nii")},
    ])
    (tmp_path / "a.nii").write_bytes(b"x")
    (tmp_path / "b.nii").write_bytes(b"x")
    cache = config.DatasetJsonCache()
    entries = cache.load_entries(str(json_path))
    assert "segmentation" not in entries[0]
    assert "segmentation" not in entries[1]


def test_validate_rejects_zero_dataset_weight():
    cfg = _minimal_config()
    cfg["datasets"][0]["weight"] = 0.0
    with pytest.raises(ValueError, match="weight"):
        config.validate_config(cfg)


def test_validate_rejects_unknown_dataset_type():
    cfg = _minimal_config()
    cfg["datasets"][0]["type"] = "not_a_type"
    with pytest.raises(ValueError, match="unknown type"):
        config.validate_config(cfg)


def test_validate_warns_on_unknown_training_key(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        config.validate_config(_minimal_config(some_typo_key=42))
    assert any("Unrecognized training keys" in r.message for r in caplog.records)


def _write_json(tmp_path, name, data):
    path = tmp_path / name
    with open(path, "w") as f:
        json.dump({"data": data}, f)
    return path


def _schema_with_paired_dataset(tmp_path, json_path):
    """Build a FinetuningConfigSchema with one paired dataset pointing at
    the given JSON. Bypasses validate_config to focus on the paired check."""
    return config.FinetuningConfigSchema.from_dict({
        "experiment": {"name": "demo", "model_weights": "unigradicon"},
        "datasets": [{"name": "d1", "type": "paired", "json_file": str(json_path)}],
    })


def test_paired_check_raises_when_no_subject_id_anywhere(tmp_path):
    json_path = _write_json(tmp_path, "data.json", [
        {"image": "a.nii"}, {"image": "b.nii"},
    ])
    schema = _schema_with_paired_dataset(tmp_path, json_path)
    cache = config.DatasetJsonCache()
    with pytest.raises(ValueError, match="paired"):
        config.validate_paired_datasets_have_pairs(schema, str(tmp_path), cache)


def test_paired_check_raises_when_each_subject_has_only_one_image(tmp_path):
    json_path = _write_json(tmp_path, "data.json", [
        {"image": "a.nii", "subject_id": "s1"},
        {"image": "b.nii", "subject_id": "s2"},
    ])
    schema = _schema_with_paired_dataset(tmp_path, json_path)
    cache = config.DatasetJsonCache()
    with pytest.raises(ValueError, match="paired"):
        config.validate_paired_datasets_have_pairs(schema, str(tmp_path), cache)


def test_paired_check_passes_when_at_least_one_subject_has_two_images(tmp_path):
    json_path = _write_json(tmp_path, "data.json", [
        {"image": "a.nii", "subject_id": "s1"},
        {"image": "b.nii", "subject_id": "s1"},
        {"image": "c.nii", "subject_id": "s2"},
    ])
    schema = _schema_with_paired_dataset(tmp_path, json_path)
    cache = config.DatasetJsonCache()
    config.validate_paired_datasets_have_pairs(schema, str(tmp_path), cache)


def test_paired_check_skips_unpaired_datasets(tmp_path):
    """An 'unpaired' dataset with no subject_id should not trigger the check."""
    json_path = _write_json(tmp_path, "data.json", [
        {"image": "a.nii"}, {"image": "b.nii"},
    ])
    schema = config.FinetuningConfigSchema.from_dict({
        "experiment": {"name": "demo", "model_weights": "unigradicon"},
        "datasets": [{"name": "d1", "type": "unpaired", "json_file": str(json_path)}],
    })
    cache = config.DatasetJsonCache()
    config.validate_paired_datasets_have_pairs(schema, str(tmp_path), cache)


def test_required_fields_empty_for_image_only_training():
    tc = config.TrainingConfig()
    assert config.required_data_fields(tc) == frozenset()


def test_required_fields_includes_segmentation_when_dice_loss_set():
    tc = config.TrainingConfig.from_dict({"dice_loss_weight": 0.5})
    assert "segmentation" in config.required_data_fields(tc)


def test_required_fields_includes_mask_when_loss_function_masking():
    tc = config.TrainingConfig.from_dict({"loss_function_masking": True})
    assert "mask" in config.required_data_fields(tc)


def test_required_fields_includes_mask_when_roi_masking():
    tc = config.TrainingConfig.from_dict({"roi_masking": True})
    assert "mask" in config.required_data_fields(tc)
