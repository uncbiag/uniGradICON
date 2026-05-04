"""Unit tests for the pair samplers in unigradicon.finetuning.dataset."""
import random
import pytest

from unigradicon.finetuning import dataset


def test_random_pair_sampler_rejects_short_keys():
    with pytest.raises(ValueError, match="at least 2 keys"):
        dataset.RandomPairSampler(["only_one"])


def test_random_pair_sampler_partner_is_never_anchor():
    keys = ["k0", "k1", "k2", "k3", "k4"]
    sampler = dataset.RandomPairSampler(keys)
    random.seed(0)
    for _ in range(500):
        for anchor in keys:
            partner = sampler.sample_partner(anchor)
            assert partner != anchor
            assert partner in keys


def test_random_pair_sampler_full_coverage():
    """Every non-anchor key must be reachable as a partner for every anchor."""
    keys = ["k0", "k1", "k2", "k3"]
    sampler = dataset.RandomPairSampler(keys)
    random.seed(0)
    seen = {a: set() for a in keys}
    for _ in range(2000):
        for a in keys:
            seen[a].add(sampler.sample_partner(a))
    for a in keys:
        assert seen[a] == set(keys) - {a}, (
            f"anchor {a} did not reach every other key: {seen[a]}"
        )


def test_random_pair_sampler_uniformity():
    """Each non-anchor key should be picked with equal probability."""
    keys = ["k0", "k1", "k2", "k3", "k4"]
    sampler = dataset.RandomPairSampler(keys)
    random.seed(0)
    counts = {k: 0 for k in keys}
    n = 40000
    for _ in range(n):
        counts[sampler.sample_partner("k2")] += 1
    assert counts["k2"] == 0
    expected = n / 4
    for k in ("k0", "k1", "k3", "k4"):
        ratio = counts[k] / expected
        assert 0.95 < ratio < 1.05, f"non-uniform: {k} ratio {ratio}"


def _entries(*pairs):
    """Build a DatasetEntry list from ``(image_path, subject_id)`` pairs."""
    return [dataset.DatasetEntry(image=p, subject_id=s) for p, s in pairs]


def test_subject_pair_sampler_rejects_when_no_subject_has_pair():
    entries = _entries(("k0", "s1"), ("k1", "s2"), ("k2", "s3"))
    with pytest.raises(ValueError, match="no valid pairs"):
        dataset.SubjectPairSampler(entries, [e.image for e in entries], "demo")


def test_subject_pair_sampler_partner_respects_subject():
    entries = _entries(
        ("k0", "s1"), ("k1", "s1"), ("k2", "s1"),
        ("k3", "s2"), ("k4", "s2"),
        ("k5", "s3"),  # orphan subject; k5 must not appear as a sampler key.
    )
    keys = [e.image for e in entries]
    sampler = dataset.SubjectPairSampler(entries, keys, "demo")

    assert "k5" not in sampler.keys  # orphan filtered out
    random.seed(0)
    for _ in range(50):
        for anchor in ("k0", "k1", "k2"):
            partner = sampler.sample_partner(anchor)
            assert partner in {"k0", "k1", "k2"} - {anchor}
        for anchor in ("k3", "k4"):
            partner = sampler.sample_partner(anchor)
            assert partner in {"k3", "k4"} - {anchor}


def test_subject_pair_sampler_filters_keys_without_pairs():
    """Keys missing from the entry list (or without subject_id) are dropped."""
    entries = _entries(("k0", "s1"), ("k1", "s1"))
    # k2 is in the keys list but has no entry/subject_id mapping.
    sampler = dataset.SubjectPairSampler(entries, ["k0", "k1", "k2"], "demo")
    assert sampler.keys == ["k0", "k1"]
