import footsteps
import icon_registration as icon
import icon_registration.itk_wrapper as itk_wrapper
import itk
import numpy as np
import torch

import sys
sys.path.append("/playpen-raid2/lin.tian/projects/uniGradICON/evaluations")
import utils
import argparse

from unigradicon import make_network
from model_inshape import inshape_dict


def itk_half_scale_image(img):
    scale = 0.5
    input_size = itk.size(img)
    input_spacing = itk.spacing(img)
    input_origin = itk.origin(img)
    dimension = img.GetImageDimension()

    output_size = [int(input_size[d] * scale) for d in range(dimension)]
    output_spacing = [input_spacing[d] / scale for d in range(dimension)]
    output_origin = [
        input_origin[d] + 0.5 * (output_spacing[d] - input_spacing[d])
        for d in range(dimension)
    ]

    interpolator = itk.NearestNeighborInterpolateImageFunction.New(img)

    resampled = itk.resample_image_filter(
        img,
        transform=itk.IdentityTransform[itk.D, 3].New(),
        interpolator=interpolator,
        size=output_size,
        output_spacing=output_spacing,
        output_origin=output_origin,
        output_direction=img.GetDirection(),
    )
    # print(img)
    # print(resampled)
    # exit()

    return resampled

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
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/OAI")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

input_shape = inshape_dict[f"{args.model_type}_model"]

net = make_network(
    input_shape, include_last_step=True#, lmbda=0.2, loss_fn=icon.ssd_only_interpolated
)


logger.log(net.regis_net.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True))
net.eval()

with open("/playpen-raid2/lin.tian/projects/icon_lung/ICON/training_scripts/oai_paper_pipeline/splits/test/pair_path_list.txt") as f:
    test_pair_paths = f.readlines()

dices = []
flips = []

for test_pair_path in test_pair_paths:
    test_pair_path = test_pair_path.replace("playpen", "playpen-raid").split()
    test_pair = [itk.imread(path) for path in test_pair_path]
    test_pair = [
        (
            itk.flip_image_filter(t, flip_axes=(False, False, True))
            if "RIGHT" in path
            else t
        )
        for (t, path) in zip(test_pair, test_pair_path)
    ]
    image_A, image_B, segmentation_A, segmentation_B = test_pair

    segmentation_A = itk_half_scale_image(segmentation_A)
    segmentation_B = itk_half_scale_image(segmentation_B)

    phi_AB, phi_BA, loss = itk_wrapper.register_pair(
        net, image_A, image_B, finetune_steps=None if args.io_steps == 0 else args.io_steps, return_artifacts=True
    )

    interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_A)

    warped_segmentation_A = itk.resample_image_filter(
        segmentation_A,
        transform=phi_AB,
        interpolator=interpolator,
        use_reference_image=True,
        reference_image=segmentation_B,
    )
    mean_dice = utils.itk_mean_dice(segmentation_B, warped_segmentation_A)

    logger.log(mean_dice)
    logger.log(icon.losses.to_floats(loss))
    flips.append(loss.flips)

    dices.append(mean_dice)

logger.log("Mean DICE")
logger.log(np.mean(dices))
logger.log("flips percentage:", np.mean(flips) / np.prod(input_shape) * 100)
