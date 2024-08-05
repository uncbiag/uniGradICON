import itk
import numpy as np
import unittest
import icon_registration.test_utils

import subprocess
import os
import torch


class TestCommandInterface(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        icon_registration.test_utils.download_test_data()
        self.test_data_dir = icon_registration.test_utils.TEST_DATA_DIR
        self.test_temp_dir = f"{self.test_data_dir}/temp"
        os.makedirs(self.test_temp_dir, exist_ok=True)
        self.device = torch.cuda.current_device()

    def test_register_unigradicon_inference(self):
        subprocess.run([
            "unigradicon-register",
            "--fixed", f"{self.test_data_dir}/lung_test_data/copd1_highres_EXP_STD_COPD_img.nii.gz",
            "--fixed_modality", "ct",
            "--fixed_segmentation", f"{self.test_data_dir}/lung_test_data/copd1_highres_EXP_STD_COPD_label.nii.gz",
            "--moving", f"{self.test_data_dir}/lung_test_data/copd1_highres_INSP_STD_COPD_img.nii.gz",
            "--moving_modality", "ct",
            "--moving_segmentation", f"{self.test_data_dir}/lung_test_data/copd1_highres_INSP_STD_COPD_label.nii.gz",
            "--transform_out", f"{self.test_temp_dir}/transform.hdf5",
            "--io_iterations", "None"
        ])

        # load transform
        phi_AB = itk.transformread(f"{self.test_temp_dir}/transform.hdf5")[0]

        assert isinstance(phi_AB, itk.CompositeTransform)

        insp_points = icon_registration.test_utils.read_copd_pointset(
            str(
                icon_registration.test_utils.TEST_DATA_DIR
                / "lung_test_data/copd1_300_iBH_xyz_r1.txt"
            )
        )
        exp_points = icon_registration.test_utils.read_copd_pointset(
            str(
                icon_registration.test_utils.TEST_DATA_DIR
                / "lung_test_data/copd1_300_eBH_xyz_r1.txt"
            )
        )

        dists = []
        for i in range(len(insp_points)):
            px, py = (
                insp_points[i],
                np.array(phi_AB.TransformPoint(tuple(exp_points[i]))),
            )
            dists.append(np.sqrt(np.sum((px - py) ** 2)))
        print(np.mean(dists))
        self.assertLess(np.mean(dists), 2.1)

        # remove temp file
        os.remove(f"{self.test_temp_dir}/transform.hdf5")
    
    def test_register_multigradicon_inference(self):
        subprocess.run([
            "unigradicon-register",
            "--fixed", f"{self.test_data_dir}/lung_test_data/copd1_highres_EXP_STD_COPD_img.nii.gz",
            "--fixed_modality", "ct",
            "--fixed_segmentation", f"{self.test_data_dir}/lung_test_data/copd1_highres_EXP_STD_COPD_label.nii.gz",
            "--moving", f"{self.test_data_dir}/lung_test_data/copd1_highres_INSP_STD_COPD_img.nii.gz",
            "--moving_modality", "ct",
            "--moving_segmentation", f"{self.test_data_dir}/lung_test_data/copd1_highres_INSP_STD_COPD_label.nii.gz",
            "--transform_out", f"{self.test_temp_dir}/transform.hdf5",
            "--io_iterations", "None",
            "--model", "multigradicon"
        ])

        # load transform
        phi_AB = itk.transformread(f"{self.test_temp_dir}/transform.hdf5")[0]

        assert isinstance(phi_AB, itk.CompositeTransform)

        insp_points = icon_registration.test_utils.read_copd_pointset(
            str(
                icon_registration.test_utils.TEST_DATA_DIR
                / "lung_test_data/copd1_300_iBH_xyz_r1.txt"
            )
        )
        exp_points = icon_registration.test_utils.read_copd_pointset(
            str(
                icon_registration.test_utils.TEST_DATA_DIR
                / "lung_test_data/copd1_300_eBH_xyz_r1.txt"
            )
        )

        dists = []
        for i in range(len(insp_points)):
            px, py = (
                insp_points[i],
                np.array(phi_AB.TransformPoint(tuple(exp_points[i]))),
            )
            dists.append(np.sqrt(np.sum((px - py) ** 2)))
        print(np.mean(dists))
        self.assertLess(np.mean(dists), 3.8)

        # remove temp file
        os.remove(f"{self.test_temp_dir}/transform.hdf5")

        
        
