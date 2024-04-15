import itk
import numpy as np
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


import icon_registration.test_utils
import icon_registration.itk_wrapper
import icon_registration.pretrained_models

from unigradicon import preprocess, get_unigradicon


class TestItkInterface(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        icon_registration.test_utils.download_test_data()
        self.test_data_dir = icon_registration.test_utils.TEST_DATA_DIR


    def test_register_pair(self):
        fixed_path = f"{self.test_data_dir}/brain_test_data/8_T1w_acpc_dc_restore_brain.nii.gz"
        moving_path = f"{self.test_data_dir}/brain_test_data/2_T1w_acpc_dc_restore_brain.nii.gz"

        # Run ITK interface
        fixed = itk.imread(fixed_path)
        moving = itk.imread(moving_path)

        net = get_unigradicon()

        phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(
            net,
            preprocess(moving, "mri"), 
            preprocess(fixed, "mri"), 
            finetune_steps=None)

        phi_AB_vector = net.phi_AB_vectorfield

        # Compute the reference
        def preprocess_in_torch(img):
            im_min, im_max = torch.min(img), np.quantile(img.numpy().flatten(), 0.99) #torch.quantile(img.view(-1), 0.99)
            img = torch.clip(img, im_min, im_max)
            img = (img-im_min) / (im_max-im_min)
            return img
        
        shape = net.identity_map.shape

        fixed = torch.from_numpy(np.array(itk.imread(fixed_path), dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        fixed_in_net = preprocess_in_torch(fixed)
        fixed_in_net = F.interpolate(fixed_in_net, shape[2:], mode='trilinear', align_corners=False)

        moving = torch.Tensor(np.array(itk.imread(moving_path), dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        moving_in_net = preprocess_in_torch(moving)
        moving_in_net = F.interpolate(moving_in_net, shape[2:], mode='trilinear', align_corners=False)

        net = get_unigradicon()
        with torch.no_grad():
            net(moving_in_net.cuda(), fixed_in_net.cuda())

        self.assertLess(
            torch.mean(torch.abs(phi_AB_vector - net.phi_AB_vectorfield)), 1e-5
        )


    def test_preprocessing_mri(self):
        img_path = f"{self.test_data_dir}/brain_test_data/8_T1w_acpc_dc_restore_brain.nii.gz"

        # Run ITK interface
        img = itk.imread(img_path)
        img_preprocessed = preprocess(img, "mri")

        # Compute the reference
        def preprocess_in_torch(img):
            im_min, im_max = torch.min(img), np.quantile(img.numpy().flatten(), 0.99) #torch.quantile(img.view(-1), 0.99)
            img = torch.clip(img, im_min, im_max)
            img = (img-im_min) / (im_max-im_min)
            return img
        reference = preprocess_in_torch(torch.Tensor(np.array(img, dtype=np.float32))).numpy()

        self.assertLess(
            np.mean(np.abs(reference - img_preprocessed)), 1e-5
        )
    
    def test_preprocessing_ct(self):
        img_path = f"{self.test_data_dir}/lung_test_data/copd1_highres_EXP_STD_COPD_img.nii.gz"

        # Run ITK interface
        img = itk.imread(img_path)
        img_preprocessed = preprocess(img, "ct")

        # Compute the reference
        def preprocess_in_torch(img):
            im_min, im_max = -1000, 1000
            img = torch.clip(img, im_min, im_max)
            img = (img-im_min) / (im_max-im_min)
            return img
        reference = preprocess_in_torch(torch.Tensor(np.array(img, dtype=np.float32))).numpy()

        self.assertLess(
            np.mean(np.abs(reference - img_preprocessed)), 1e-5
        )
    
    def test_itk_registration(self):
        net = get_unigradicon()
        
        image_exp = itk.imread(
            str(
                self.test_data_dir
                / "lung_test_data/copd1_highres_EXP_STD_COPD_img.nii.gz"
            )
        )
        image_insp = itk.imread(
            str(
                self.test_data_dir
                / "lung_test_data/copd1_highres_INSP_STD_COPD_img.nii.gz"
            )
        )
        image_exp_seg = itk.imread(
            str(
                self.test_data_dir
                / "lung_test_data/copd1_highres_EXP_STD_COPD_label.nii.gz"
            )
        )
        image_insp_seg = itk.imread(
            str(
                self.test_data_dir
                / "lung_test_data/copd1_highres_INSP_STD_COPD_label.nii.gz"
            )
        )

        image_insp_preprocessed = preprocess(image_insp, "ct", image_insp_seg)
        image_exp_preprocessed = preprocess(image_exp, "ct", image_exp_seg)

        phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(
            net, image_insp_preprocessed, image_exp_preprocessed, finetune_steps=None
        )

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
                exp_points[i],
                np.array(phi_BA.TransformPoint(tuple(insp_points[i]))),
            )
            dists.append(np.sqrt(np.sum((px - py) ** 2)))
        self.assertLess(np.mean(dists), 1.7)

        dists = []
        for i in range(len(insp_points)):
            px, py = (
                insp_points[i],
                np.array(phi_AB.TransformPoint(tuple(exp_points[i]))),
            )
            dists.append(np.sqrt(np.sum((px - py) ** 2)))
        self.assertLess(np.mean(dists), 2.1)
    
    def test_itk_warp(self):
        fixed_path = f"{self.test_data_dir}/brain_test_data/8_T1w_acpc_dc_restore_brain.nii.gz"
        moving_path = f"{self.test_data_dir}/brain_test_data/2_T1w_acpc_dc_restore_brain.nii.gz"

        # Run ITK interface
        fixed = itk.imread(fixed_path)
        moving = itk.imread(moving_path)

        net = get_unigradicon()

        phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(
            net,
            preprocess(moving, "mri"), 
            preprocess(fixed, "mri"), 
            finetune_steps=None)

        interpolator = itk.LinearInterpolateImageFunction.New(moving)
        warped_moving_image = np.array(itk.resample_image_filter(
                preprocess(moving, "mri"),
                transform=phi_AB,
                interpolator=interpolator,
                use_reference_image=True,
                reference_image=fixed
                ))
        
        reference = F.interpolate(net.warped_image_A, size=warped_moving_image.shape, mode='trilinear', align_corners=False)[0,0].cpu().numpy()

        from icon_registration.losses import NCC
        diff = NCC()(torch.Tensor(warped_moving_image).unsqueeze(0).unsqueeze(0), torch.Tensor(reference).unsqueeze(0).unsqueeze(0))
        self.assertLess(
            diff, 5e-3
        )
        
