"""Shared fixtures for the finetuning test package."""
import pytest
import torch

from unigradicon.finetuning import dataset


@pytest.fixture
def fake_image_reader(monkeypatch):
    """Replace ITK-backed ``ImageReader`` with a deterministic fake so tests
    do not need real medical-image files."""
    class FakeReader:
        def read(self, path):
            return torch.zeros(8, 8, 8)
    monkeypatch.setattr(dataset, "ImageReader", FakeReader)
    return FakeReader
