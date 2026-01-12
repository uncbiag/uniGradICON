import time

import footsteps
import icon_registration.itk_wrapper
import icon_registration.pretrained_models
import icon_registration.pretrained_models.lung_ct
import icon_registration.test_utils
import itk
import numpy as np
import torch

import sys
sys.path.append("/playpen-raid2/lin.tian/projects/uniGradICON/evaluations")
import utils

from unigradicon import make_network
from model_inshape import inshape_dict

image_root = "/playpen-raid1/lin.tian/data/lung/dirlab_highres_350"
landmark_root = "/playpen-raid1/lin.tian/data/lung/reg_lung_2d_3d_1000_dataset_4_proj_clean_bg/landmarks/"

cases = [f"copd{i}_highres" for i in range(1, 11)]



import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str, help="the path to the weights of the network")
parser.add_argument("--io_steps", type=int, default=0, help="Steps for IO")
parser.add_argument("--device", type=int, default=0, help="GPU ID.")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")
parser.add_argument("--model_type", type=str, default="lung", help="the type of the model. (lung, brain, knee)}")

args = parser.parse_args()
weights_path = args.weights_path
device = torch.device(f'cuda:{args.device}')
torch.cuda.set_device(device)

if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/COPDGene")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

input_shape = inshape_dict[f"{args.model_type}_model"]
net = make_network(input_shape, include_last_step=True)


logger.log(net.regis_net.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False))
net.eval()

overall_1 = []
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
        finetune_steps=None if args.io_steps == 0 else args.io_steps,
        return_artifacts=True,
    )
    torch.cuda.synchronize()
    end = time.time()
    elapsed_times.append(end - start)

    dists = []
    for i in range(len(landmarks_exp)):
        px, py = (
            landmarks_insp[i],
            np.array(phi_AB.TransformPoint(tuple(landmarks_exp[i]))),
        )
        dists.append(np.sqrt(np.sum((px - py) ** 2)))
    logger.log(f"Mean error on {case}: ", np.mean(dists))
    overall_1.append(np.mean(dists))
    dists = []
    for i in range(len(landmarks_insp)):
        px, py = (
            landmarks_exp[i],
            np.array(phi_BA.TransformPoint(tuple(landmarks_insp[i]))),
        )
        dists.append(np.sqrt(np.sum((px - py) ** 2)))
    logger.log(f"Mean error on {case}: ", np.mean(dists))

    overall_2.append(np.mean(dists))

    logger.log("flips:", loss.flips)
    logger.log("Elapsed time:", elapsed_times[-1])

    flips.append(loss.flips)


logger.log("overall:")
logger.log(np.mean(overall_1))
logger.log(np.mean(overall_2))
logger.log("flips:", np.mean(flips))
logger.log("flips percentage:", np.mean(flips) / np.prod(input_shape) * 100)

logger.log("Elaspsed time.", np.mean(elapsed_times))
