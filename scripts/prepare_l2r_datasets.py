#!/usr/bin/env python3
"""Generate JSON dataset files for Learn2Reg datasets.

Scans a directory for extracted Learn2Reg datasets (OASIS, LungCT, AbdomenMRCT)
and generates JSON files that can be used with unigradicon-finetune.

All image paths in the generated JSON files are relative to the JSON file's
directory, so the output is portable across machines.

Usage:
    python scripts/prepare_l2r_datasets.py --data_dir datasets/
"""

import argparse
import json
import os
import sys


def _rel_path(target, base_dir):
    """Return target path relative to base_dir."""
    return os.path.relpath(os.path.abspath(target), os.path.abspath(base_dir))


def _find_seg(image_path, seg_dir):
    """Find a matching segmentation file with the same basename."""
    candidate = os.path.join(seg_dir, os.path.basename(image_path))
    return candidate if os.path.exists(candidate) else None


def generate_oasis_json(data_dir, output_dir):
    """Generate JSON for OASIS brain MRI (unpaired, with segmentations).

    Uses imagesTr for images and labelsTr for 35-structure brain segmentations.
    """
    images_dir = os.path.join(data_dir, "OASIS", "imagesTr")
    labels_dir = os.path.join(data_dir, "OASIS", "labelsTr")

    if not os.path.isdir(images_dir):
        return None

    data = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.endswith(".nii.gz"):
            continue
        image_path = os.path.join(images_dir, fname)
        entry = {"image": _rel_path(image_path, output_dir)}
        if os.path.isdir(labels_dir):
            seg_path = _find_seg(image_path, labels_dir)
            if seg_path:
                entry["segmentation"] = _rel_path(seg_path, output_dir)
        data.append(entry)

    return {"data": data} if len(data) >= 2 else None


def generate_lungct_json(data_dir, output_dir):
    """Generate JSON for LungCT (paired).

    Each subject has two timepoints (_0000, _0001). Uses masksTr for binary lung masks.
    """
    images_dir = os.path.join(data_dir, "LungCT", "imagesTr")
    masks_dir = os.path.join(data_dir, "LungCT", "masksTr")

    if not os.path.isdir(images_dir):
        return None

    data = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.endswith(".nii.gz"):
            continue
        image_path = os.path.join(images_dir, fname)
        parts = fname.replace(".nii.gz", "").rsplit("_", 1)
        subject_id = parts[0]

        entry = {
            "image": _rel_path(image_path, output_dir),
            "subject_id": subject_id,
        }
        if os.path.isdir(masks_dir):
            seg_path = _find_seg(image_path, masks_dir)
            if seg_path:
                entry["mask"] = _rel_path(seg_path, output_dir)
        data.append(entry)

    return {"data": data} if len(data) >= 2 else None


def generate_abdomenmrct_json(data_dir, output_dir):
    """Generate JSON for AbdomenMRCT cross-modality dataset (unpaired, with segmentations).

    Includes all CT (_0000) and MR (_0001) images with per-image modality field.
    The dataset preprocesses each image according to its modality (CT windowing
    vs quantile normalization). Use with lncc2 for modality-invariant similarity.
    """
    images_dir = os.path.join(data_dir, "AbdomenMRCT", "imagesTr")
    labels_dir = os.path.join(data_dir, "AbdomenMRCT", "labelsTr")

    if not os.path.isdir(images_dir):
        return None

    data = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.endswith(".nii.gz"):
            continue
        image_path = os.path.join(images_dir, fname)
        modality = "ct" if fname.endswith("_0000.nii.gz") else "mri"

        entry = {
            "image": _rel_path(image_path, output_dir),
            "modality": modality,
        }
        if os.path.isdir(labels_dir):
            seg_path = _find_seg(image_path, labels_dir)
            if seg_path:
                entry["segmentation"] = _rel_path(seg_path, output_dir)
        data.append(entry)

    return {"data": data} if len(data) >= 2 else None


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON dataset files for Learn2Reg datasets"
    )
    parser.add_argument(
        "--data_dir", type=str, default="datasets",
        help="Directory containing extracted Learn2Reg datasets (default: datasets/)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to write JSON files (default: configs/learn2reg/)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found: {os.path.abspath(data_dir)}")
        print("Download and extract Learn2Reg datasets first. See the finetuning README.")
        sys.exit(1)

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "configs", "learn2reg")
    else:
        output_dir = args.output_dir
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    generators = [
        ("l2r_oasis.json", "OASIS", generate_oasis_json),
        ("l2r_lungct.json", "LungCT", generate_lungct_json),
        ("l2r_abdomenmrct.json", "AbdomenMRCT", generate_abdomenmrct_json),
    ]

    print(f"Scanning {os.path.abspath(data_dir)} for Learn2Reg datasets...\n")
    generated = []
    for json_name, dataset_name, generator in generators:
        dataset_path = os.path.join(data_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            print(f"  {dataset_name}: not found, skipping")
            continue

        result = generator(data_dir, output_dir)
        if result is None:
            print(f"  {dataset_name}: found but too few images, skipping")
            continue

        output_path = os.path.join(output_dir, json_name)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        has_seg = any("segmentation" in entry for entry in result["data"])
        has_mask = any("mask" in entry for entry in result["data"])
        extras = []
        if has_seg:
            extras.append("segmentations")
        if has_mask:
            extras.append("masks")
        extra_info = f" (with {', '.join(extras)})" if extras else ""
        print(f"  {dataset_name}: {len(result['data'])} images{extra_info} -> {json_name}")
        generated.append(json_name)

    if not generated:
        print("\nNo datasets found. Expected structure:")
        print(f"  {data_dir}/OASIS/imagesTr/*.nii.gz")
        print(f"  {data_dir}/LungCT/imagesTr/*.nii.gz")
        print(f"  {data_dir}/AbdomenMRCT/imagesTr/*.nii.gz")
        print("\nDownload from: https://learn2reg.grand-challenge.org/Datasets/")
        sys.exit(1)

    print(f"\nGenerated {len(generated)} JSON file(s) in {output_dir}")
    print("All paths are relative — portable across machines.\n")
    print("Run finetuning with:")
    for name in generated:
        config_name = name.replace(".json", ".yaml")
        print(f"  unigradicon-finetune --config configs/learn2reg/{config_name}")
    if len(generated) > 1:
        print(f"  unigradicon-finetune --config configs/learn2reg/l2r_multi.yaml")


if __name__ == "__main__":
    main()
