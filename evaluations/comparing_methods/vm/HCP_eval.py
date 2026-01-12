import footsteps
import random
import os
import argparse
import pandas as pd
import json

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str, help="the path to the weights of the network")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")
parser.add_argument("--device", type=int, default=0, help="GPU ID.")
args = parser.parse_args()

weights_path = args.weights_path

import torch
import itk
import numpy as np
import icon_registration.itk_wrapper as itk_wrapper
from unigradicon.train import make_network
import utils
# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm 


def preprocess(image):
    #image = itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.F, 3]].New()(image)
    max_ = np.max(np.array(image))
    image = itk.shift_scale_image_filter(image, shift=0., scale = .9 / max_)
    
    #image = itk.clamp_image_filter(image, bounds=(0, 1))
    return image


input_shape = [1, 1, 176, 176, 176]
device = torch.device(f'cuda:{args.device}')
torch.cuda.set_device(device)

if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/HCP")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

# Save the registration results
result_save_dir = os.path.join(footsteps.output_dir, "results")
os.makedirs(result_save_dir, exist_ok=True)

net = vxm.networks.VxmDense.load(weights_path, device)
net.eval()

dices = []
flips = []
shape_prod = np.prod(input_shape)

from HCP_segs import (atlas_registered, get_sub_seg, get_brain_image)

pair_list = []
results = []
random.seed(1)
for _ in range(100):
    n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
    image_A, image_B = (preprocess(get_brain_image(n)) for n in (n_A, n_B))

    case_id = f"HCP.{n_A}_to_{n_B}"
    case_result_save_dir = os.path.join(result_save_dir, case_id)
    os.makedirs(case_result_save_dir, exist_ok=True)

    #import pdb; pdb.set_trace()
    phi_AB, phi_BA, flip, phi_AB_vectorfield, phi_BA_vectorfield= utils.register_pair(
        net, image_A, image_B, model_input_shape=input_shape, device=device
    )

    segmentation_A, segmentation_B = (get_sub_seg(n) for n in (n_A, n_B))

    interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_A)

    warped_segmentation_A = itk.resample_image_filter(
            segmentation_A, 
            transform=phi_AB,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=segmentation_B
            )
    
    # Save the registration results
    itk.imwrite(segmentation_A, os.path.join(case_result_save_dir, f"{case_id}.{n_A}_seg.nii.gz"))
    itk.imwrite(segmentation_B, os.path.join(case_result_save_dir, f"{case_id}.{n_B}_seg.nii.gz"))
    itk.imwrite(warped_segmentation_A, os.path.join(case_result_save_dir, f"{case_id}.warped_{n_A}_seg.nii.gz"))

    # Compute the metrics
    flips_percentage, log_jac_std = utils.compute_jacob_det_from_itk_transform(phi_AB, segmentation_B)
    flips_percentage_in_img, log_jac_std_in_img = utils.compute_jacob_det(phi_AB_vectorfield)

    metric = utils.compute_metrics(
        segmentation_B, segmentation_A, warped_segmentation_A, labels=utils.get_label_list(case_id))

    results.append(
        (
            case_id,
            os.path.join(case_result_save_dir, f"{case_id}.{n_A}_seg.nii.gz"),
            os.path.join(case_result_save_dir, f"{case_id}.{n_B}_seg.nii.gz"),
            os.path.join(case_result_save_dir, f"{case_id}.warped_{n_A}_seg.nii.gz"),
            flips_percentage,
            log_jac_std,
            flips_percentage_in_img,
            log_jac_std_in_img,
            metric["dice"],
            metric["hd95"]
        )
    )
    pair_list.append([n_A, n_B])
    # mean_dice = utils.itk_mean_dice(segmentation_B, warped_segmentation_A)
    
    # flips.append(flip)
    # dices.append(mean_dice)
    # utils.log(f"{_}/100 mean DICE {n_A} to {n_B}: {mean_dice} | running DICE: {np.mean(dices)} | Percentage: {flip} | running Per: {np.mean(flips)} ")
    # pair_list.append([n_A, n_B])

with open(f'{footsteps.output_dir}/pair_list.txt', 'w') as f:
    for p in pair_list:
        f.write(",".join(p)+'\n')

df = pd.DataFrame(results, columns=[
    "case_id", "segmentation_A", 
    "segmentation_B", "warped_segmentation_A", 
    "flips_percentage_itk", "log_jac_std_itk",
    "flips_percentage", "log_jac_std",
    "dice", "hd95"])
df.to_csv(os.path.join(footsteps.output_dir, "results.csv"), index=False)

# Aggregate the metrics
dices = [dice*100. for dice in df["dice"].tolist()]
aggregate = {
    "dice": {"mean": np.mean(dices), "std": np.std(dices)},
    "hd95": {"mean": np.mean(df["hd95"]), "std": np.std(df["hd95"])},
    "flips_percentage_itk": {"mean": np.mean(df["flips_percentage_itk"]), "std": np.std(df["flips_percentage_itk"])},
    "log_jac_std_itk": {"mean": np.mean(df["log_jac_std_itk"]), "std": np.std(df["log_jac_std_itk"])},
    "flips_percentage": {"mean": np.mean(df["flips_percentage"]), "std": np.std(df["flips_percentage"])},
    "log_jac_std": {"mean": np.mean(df["log_jac_std"]), "std": np.std(df["log_jac_std"])}
}
logger.log("Aggregate:")
for k, v in aggregate.items():
    logger.log(f"{k}: {v['mean']:.2f} ± {v['std']:.2f}")

def convert_numpy_float(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)


with open(os.path.join(footsteps.output_dir, "aggregate_results.json"), 'w') as f:
    json.dump(aggregate, f, default=convert_numpy_float, indent=4)

logger.close()
