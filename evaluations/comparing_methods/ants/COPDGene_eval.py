import os
import time
import footsteps
import icon_registration.itk_wrapper
import icon_registration.pretrained_models
import icon_registration.pretrained_models.lung_ct
import icon_registration.test_utils
import itk
import numpy as np
import torch
from ants_helper import compute_jacob_det_for_ants, copy_reference_image_info, run_ants

import sys
# Create the absolute path by joining the current directory and the relative path
absolute_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(absolute_path)
import utils

image_root = "/playpen-raid1/lin.tian/data/lung/dirlab_highres_350"
landmark_root = "/playpen-raid1/lin.tian/data/lung/reg_lung_2d_3d_1000_dataset_4_proj_clean_bg/landmarks/"

cases = [f"copd{i}_highres" for i in range(1, 11)]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, default="", help="Experiment name.")

args = parser.parse_args()
if args.exp == "":
    footsteps.initialize(output_root="/playpen-ssd/lin.tian/results/unigradicon/comparing_methods/ants/")
else:
    footsteps.initialize(output_root="/playpen-ssd/lin.tian/results/unigradicon/comparing_methods/ants/", run_name=f"{args.exp}/HCP")

os.makedirs(f"{footsteps.output_dir}/tmp", exist_ok=True)

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")


overall_2 = []
flips = []
elapsed_times = []

for case in cases:
    image_insp = itk.imread(f"{image_root}/{case}/{case}_INSP_STD_COPD_img.nii.gz")
    image_exp = itk.imread(f"{image_root}/{case}/{case}_EXP_STD_COPD_img.nii.gz")
    seg_insp = itk.imread(f"{image_root}/{case}/{case}_INSP_STD_COPD_label.nii.gz")
    seg_exp = itk.imread(f"{image_root}/{case}/{case}_EXP_STD_COPD_label.nii.gz")

    landmarks_insp = icon_registration.test_utils.read_copd_pointset(
        landmark_root + f"/{case.split('_')[0]}_300_iBH_xyz_r1.txt"
    )
    landmarks_exp = icon_registration.test_utils.read_copd_pointset(
        landmark_root + f"/{case.split('_')[0]}_300_eBH_xyz_r1.txt"
    )

    image_insp_preprocessed = (
        icon_registration.pretrained_models.lung_network_preprocess(
            image_insp, seg_insp
        )
    )
    image_exp_preprocessed = (
        icon_registration.pretrained_models.lung_network_preprocess(image_exp, seg_exp)
    )

    torch.cuda.synchronize()
    start = time.time()
    phi_AB, phi_BA, loss = icon_registration.itk_wrapper.register_pair(
        net,
        image_insp_preprocessed,
        image_exp_preprocessed,
        finetune_steps=None,
        return_artifacts=True,
    )
    torch.cuda.synchronize()
    end = time.time()
    elapsed_times.append(end - start)


    dists = []
    for i in range(len(landmarks_insp)):
        px, py = (
            landmarks_exp[i],
            np.array(phi_BA.TransformPoint(tuple(landmarks_insp[i]))),
        )
        dists.append(np.sqrt(np.sum((px - py) ** 2)))
    utils.log(f"Mean error on {case}: ", np.mean(dists))

    overall_2.append(np.mean(dists))

    utils.log("flips:", loss.flips)
    utils.log("Elapsed time:", elapsed_times[-1])

    flips.append(loss.flips)


utils.log("overall:")
utils.log(np.mean(overall_1))
utils.log(np.mean(overall_2))
utils.log("flips:", np.mean(flips))
utils.log("flips / prod(imnput_shape", np.mean(flips) / np.prod(input_shape))

utils.log("Elaspsed time.", np.mean(elapsed_times))
