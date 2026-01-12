import footsteps
import os

import icon_registration as icon
import icon_registration.losses as losses
from unigradicon.train import make_network
import torch
import itk
import numpy as np
import icon_registration.itk_wrapper as itk_wrapper
import utils
import pandas as pd
import json

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm 


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


input_shape = [1, 1, 176, 176, 176]


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str, help="the path to the weights of the network")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")
parser.add_argument("--device", type=int, default=0, help="GPU ID.")

args = parser.parse_args()
weights_path = args.weights_path
device = torch.device(f'cuda:{args.device}')
torch.cuda.set_device(device)

if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/OAI")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

# Save the registration results
result_save_dir = os.path.join(footsteps.output_dir, "results")
os.makedirs(result_save_dir, exist_ok=True)

net = vxm.networks.VxmDense.load(weights_path, device)
net.eval()

with open("/playpen-raid2/lin.tian/projects/icon_lung/ICON/training_scripts/oai_paper_pipeline/splits/test/pair_path_list.txt") as f:
    test_pair_paths = f.readlines()


results = []
for test_pair_path in test_pair_paths:
    test_pair_path = test_pair_path.replace("playpen", "playpen-raid").split()

    case_id_A = test_pair_path[0].split("/")[-1].split(".")[0]
    case_id_B = test_pair_path[1].split("/")[-1].split(".")[0]
    case_id = f"OAI.{case_id_A}_to_{case_id_B}"
    case_result_save_dir = os.path.join(result_save_dir, case_id)
    os.makedirs(case_result_save_dir, exist_ok=True)

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

    phi_AB, phi_BA, flip, phi_AB_vectorfield, phi_BA_vectorfield = utils.register_pair(
        net, image_A, image_B, model_input_shape=input_shape, device=device
    )

    interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_A)

    warped_segmentation_A = itk.resample_image_filter(
        segmentation_A,
        transform=phi_AB,
        interpolator=interpolator,
        use_reference_image=True,
        reference_image=segmentation_B,
    )
    
    # Save the registration results
    itk.imwrite(segmentation_A, os.path.join(case_result_save_dir, f"{case_id}.{case_id_A}_seg.nii.gz"))
    itk.imwrite(segmentation_B, os.path.join(case_result_save_dir, f"{case_id}.{case_id_B}_seg.nii.gz"))
    itk.imwrite(warped_segmentation_A, os.path.join(case_result_save_dir, f"{case_id}.warped_{case_id_A}_seg.nii.gz"))

    # Compute the metrics
    flips_percentage, log_jac_std = utils.compute_jacob_det_from_itk_transform(phi_AB, segmentation_B)
    flips_percentage_in_img, log_jac_std_in_img = utils.compute_jacob_det(phi_AB_vectorfield)

    metric = utils.compute_metrics(
        segmentation_B, segmentation_A, warped_segmentation_A, labels=utils.get_label_list(case_id))

    results.append(
        (
            case_id,
            os.path.join(case_result_save_dir, f"{case_id}.{case_id_A}_seg.nii.gz"),
            os.path.join(case_result_save_dir, f"{case_id}.{case_id_B}_seg.nii.gz"),
            os.path.join(case_result_save_dir, f"{case_id}.warped_{case_id_A}_seg.nii.gz"),
            flips_percentage,
            log_jac_std,
            flips_percentage_in_img,
            log_jac_std_in_img,
            metric["dice"],
            metric["hd95"]
        )
    )

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
