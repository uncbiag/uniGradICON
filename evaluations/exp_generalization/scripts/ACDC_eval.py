import argparse
import random

import footsteps
import icon_registration.itk_wrapper as itk_wrapper
import itk
import numpy as np
import torch
import glob
import os
import configparser
import pandas as pd

import sys
sys.path.append("/playpen-raid2/lin.tian/projects/uniGradICON/evaluations")
import utils

from unigradicon import make_network
from model_inshape import inshape_dict

def quantile(arr: torch.Tensor, q):
    arr = arr.flatten()
    l = len(arr)
    return torch.kthvalue(arr, int(q * l)).values

def preprocess(image):
    image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
    min_, _ = itk.image_intensity_min_max(image)
    max_ = quantile(torch.tensor(np.array(image)), .99).item()
    image = itk.clamp_image_filter(image, Bounds=(min_, max_))
    image = itk.shift_scale_image_filter(image, shift=-min_, scale = 1/(max_-min_)) 
    return image

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str, help="the path to the weights of the network")
parser.add_argument("--io_steps", type=int, default=0, help="Steps for IO")
parser.add_argument("--device", type=int, default=0, help="GPU ID.")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")
parser.add_argument("--bidirection", type=int, default=0, help="Whether to evaluate the registration in two directions.")
parser.add_argument("--model_type", type=str, default="lung", help="the type of the model. (lung, brain, knee)}")

args = parser.parse_args()
weights_path = args.weights_path
device = torch.device(f'cuda:{args.device}')
torch.cuda.set_device(device)

bidirection = args.bidirection

if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/ACDC")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

input_shape = inshape_dict[f"{args.model_type}_model"] if f"{args.model_type}_model" in inshape_dict else [1, 1, 175, 175, 175]
net = make_network(input_shape, include_last_step=True)#, framework="svf")


logger.log(net.regis_net.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False))
net.eval()

dices = []
dices_origin = []
flips = []
if bidirection:
    dices_inverse = []
    flips_inverse = []
shape_prod = np.prod(input_shape)

test_cases = sorted(glob.glob('/playpen-raid/bdemir/uniICON/ACDC/database/testing/patient*/'))
config = configparser.ConfigParser()

for case in test_cases:
    case_id = case.split("/")[-2]
    image_A = preprocess(itk.imread(os.path.join(case, f"{case_id}_frame01.nii.gz")))
    with open(os.path.join(case, "Info.cfg")) as f:
        image_B_id = int(f.readlines()[1].split(":")[-1][:-1])
    image_B = preprocess(itk.imread(os.path.join(case, f"{case_id}_frame{image_B_id:02d}.nii.gz")))

    phi_AB, phi_BA, loss= itk_wrapper.register_pair(net, image_A, image_B, finetune_steps=None if args.io_steps == 0 else args.io_steps,
        return_artifacts=True,
    )

    segmentation_A = itk.imread(os.path.join(case, f"{case_id}_frame01_gt.nii.gz"))
    segmentation_B = itk.imread(os.path.join(case, f"{case_id}_frame{image_B_id:02d}_gt.nii.gz"))

    interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_A)

    warped_segmentation_A = itk.resample_image_filter(
            segmentation_A, 
            transform=phi_AB,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=segmentation_B
            )
    mean_dice = utils.itk_mean_dice(segmentation_B, warped_segmentation_A)

    if bidirection:
        # Swap the images
        phi_BA, phi_AB, loss= itk_wrapper.register_pair(net, image_B, image_A, finetune_steps=None if args.io_steps == 0 else args.io_steps,
            return_artifacts=True,
        )

        interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_B)
        warped_segmentation_B = itk.resample_image_filter(
                segmentation_B, 
                transform=phi_BA,
                interpolator=interpolator,
                use_reference_image=True,
                reference_image=segmentation_A
                )
        mean_dice_inverse = utils.itk_mean_dice(segmentation_A, warped_segmentation_B)
        dices_inverse.append(mean_dice_inverse)
        flip_inverse = loss.flips / shape_prod * 100.
        flips_inverse.append(flip_inverse)

    mean_dice_origin = utils.itk_mean_dice(segmentation_B, segmentation_A)
    
    flip = loss.flips / shape_prod * 100.
    flips.append(flip)
    dices.append(mean_dice)
    dices_origin.append(mean_dice_origin)
    logger.log(f"{case_id} mean DICE: {mean_dice} | running DICE: {np.mean(dices)} | Percentage: {flip} | running Per: {np.mean(flips)} ")
    if bidirection:
        logger.log(f"{case_id} inverse mean DICE: {mean_dice_inverse} | running DICE: {np.mean(dices_inverse)} | Percentage: {flip_inverse} | running Per: {np.mean(flips_inverse)} ")

logger.log("Mean DICE")
logger.log(f"DICE before registration: {np.mean(dices_origin)}")
logger.log(f"Final DICE: {np.mean(dices)} | final percentage of negative jacobian: {np.mean(flips)}")
if bidirection:
    logger.log(f"Final DICE inverse: {np.mean(dices_inverse)} | final percentage of negative jacobian inverse: {np.mean(flips_inverse)}")
    logger.log(f"Final DICE (w/ two directions): ({np.mean(dices+dices_inverse)} | final percentage of negative jacobian: {np.mean(flips+flips_inverse)})")

if bidirection:
    dices_all = dices + dices_inverse
    flips_all = flips + flips_inverse
    directions = ["forward"] * len(dices) + ["backward"] * len(dices_inverse)
else:
    dices_all = dices
    flips_all = flips
    directions = ["forward"] * len(dices)

df = pd.DataFrame({"DICE":dices_all, "%folds":flips_all, "direction":directions})
df.to_csv(f"{footsteps.output_dir}/output.csv", index=False)

logger.close()
