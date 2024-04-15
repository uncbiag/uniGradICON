import itk
import numpy as np
import unittest
import numpy as np
import torch
import torch.nn.functional as F


import icon_registration.test_utils
import icon_registration.itk_wrapper

from unigradicon import preprocess, get_unigradicon


class TestItkInterface(unittest.TestCase):
    def test_itk_registration(self):
        test_case_folder = "./test_files"
        fixed_path = f"{test_case_folder}/mri_a.nii.gz"
        moving_path = f"{test_case_folder}/mri_b.nii.gz"

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
        test_case_folder = "./test_files"
        img_path = f"{test_case_folder}/mri_a.nii.gz"

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
        test_case_folder = "./test_files"
        img_path = f"{test_case_folder}/ct_test.nii.gz"

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