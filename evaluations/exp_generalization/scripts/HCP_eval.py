import argparse
import random

import footsteps
import icon_registration.itk_wrapper as itk_wrapper
import itk
import numpy as np
import torch

import sys
sys.path.append("/playpen-raid2/lin.tian/projects/uniGradICON/evaluations")
import utils

from unigradicon import make_network
from model_inshape import inshape_dict

def preprocess(image):
    #image = itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.F, 3]].New()(image)
    max_ = np.max(np.array(image))
    image = itk.shift_scale_image_filter(image, shift=0., scale = .9 / max_)
    
    #image = itk.clamp_image_filter(image, bounds=(0, 1))
    return image

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
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/HCP")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

input_shape = inshape_dict[f"{args.model_type}_model"]
net = make_network(input_shape, include_last_step=True)#, framework="svf")


logger.log(net.regis_net.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False))
net.eval()

dices = []
flips = []
shape_prod = np.prod(input_shape)

from HCP_segs import atlas_registered, get_brain_image, get_sub_seg

pair_list = []
random.seed(1)
for _ in range(100):
    n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
    image_A, image_B = (preprocess(get_brain_image(n)) for n in (n_A, n_B))

    #import pdb; pdb.set_trace()
    phi_AB, phi_BA, loss= itk_wrapper.register_pair(net, image_A, image_B, finetune_steps=None if args.io_steps == 0 else args.io_steps,
        return_artifacts=True,
    )

    # print(net.warped_image_A.shape)
    
    # import shutil
    # shutil.copyfile(f"/playpen-raid2/Data/HCP/HCP_1200/{n_A}/T1w/T1w_acpc_dc_restore_brain.nii.gz", f"{footsteps.output_dir}/{n_A}_T1w_acpc_dc_restore_brain.nii.gz")
    # shutil.copyfile(f"/playpen-raid2/Data/HCP/HCP_1200/{n_B}/T1w/T1w_acpc_dc_restore_brain.nii.gz", f"{footsteps.output_dir}/{n_B}_T1w_acpc_dc_restore_brain.nii.gz")

    # interpolator = itk.LinearInterpolateImageFunction.New(image_A)
    # warped_image_A = itk.resample_image_filter(
    #         image_A,
    #         transform=phi_AB,
    #         interpolator=interpolator,
    #         size=itk.size(image_B),
    #         output_spacing=itk.spacing(image_B),
    #         output_direction=image_B.GetDirection(),
    #         output_origin=image_B.GetOrigin(),
    #     )
    # np.save(f"{footsteps.output_dir}/{n_A}_and_{n_B}_warped.np", np.array(warped_image_A))

    segmentation_A, segmentation_B = (get_sub_seg(n) for n in (n_A, n_B))

    interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_A)

    warped_segmentation_A = itk.resample_image_filter(
            segmentation_A, 
            transform=phi_AB,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=segmentation_B
            )
    mean_dice = utils.itk_mean_dice(segmentation_B, warped_segmentation_A)
    
    flip = loss.flips / shape_prod * 100.
    flips.append(flip)
    dices.append(mean_dice)
    logger.log(f"{_}/100 mean DICE {n_A} to {n_B}: {mean_dice} | running DICE: {np.mean(dices)} | Percentage: {flip} | running Per: {np.mean(flips)} ")
    pair_list.append([n_A, n_B])

logger.log("Mean DICE")
logger.log(f"Final DICE: {np.mean(dices)} | final percentage of negative jacobian: {np.mean(flips)}")

with open(f'{footsteps.output_dir}/pair_list.txt', 'w') as f:
    for p in pair_list:
        f.write(",".join(p)+'\n')
