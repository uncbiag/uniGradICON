import argparse
import json
import os

import ants
import footsteps
import icon_registration as icon
import itk
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage.interpolation import zoom as zoom
from tqdm import tqdm
from ants_helper import compute_jacob_det_for_ants, run_ants
import pandas as pd
import json

import sys
# Create the absolute path by joining the current directory and the relative path
absolute_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(absolute_path)
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, help="the path to the folder containing learn2reg NLST dataset")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")
parser.add_argument("--transform_type", type=str, default="SyN", help="Transform Type: SyN or SyNOnly")

origin_shape = [1, 1, 224, 192, 224]
clamp = [-1000, 1000]

args = parser.parse_args()
if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/{args.transform_type}/L2R_NLST")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")
# Save the registration results
result_save_dir = os.path.join(footsteps.output_dir, "results")
os.makedirs(result_save_dir, exist_ok=True)

os.makedirs(f"{footsteps.output_dir}/submission/NLST", exist_ok=True)

with open(f"{args.data_folder}/NLST_dataset.json", 'r') as data_info:
    data_info = json.loads(data_info.read())
test_cases = [[c["fixed"], c["moving"]] for c in data_info["registration_val"]]

identity = torch.from_numpy(icon.mermaidlite.identity_map_multiN(origin_shape, [1., 1., 1.]))

results = []
for (fixed_path, moving_path) in tqdm(test_cases):
    case_id = f"{fixed_path.split('_')[1]}_{moving_path.split('_')[1]}"
    case_result_save_dir = os.path.join(result_save_dir, case_id)
    os.makedirs(case_result_save_dir, exist_ok=True)
    os.makedirs(os.path.join(case_result_save_dir, "ants"), exist_ok=True)

    fixed = np.asarray(itk.imread(os.path.join(args.data_folder, fixed_path)))
    moving = np.asarray(itk.imread(os.path.join(args.data_folder, moving_path)))
    fixed_mask = np.asarray(itk.imread(os.path.join(args.data_folder, fixed_path.replace("imagesTr", "masksTr"))))
    moving_mask = np.asarray(itk.imread(os.path.join(args.data_folder, moving_path.replace("imagesTr", "masksTr"))))

    fixed = torch.Tensor(np.array(fixed))
    fixed = (torch.clamp(fixed, clamp[0], clamp[1]) - clamp[0])/(clamp[1]-clamp[0])
    fixed = (fixed * torch.Tensor(fixed_mask)).numpy()
    fixed = ants.from_numpy(fixed)
    fixed_mask = ants.from_numpy(fixed_mask.astype(float))
    
    moving = torch.Tensor(np.array(moving))
    moving = (torch.clamp(moving, clamp[0], clamp[1]) - clamp[0])/(clamp[1]-clamp[0])
    moving = (moving * torch.Tensor(moving_mask)).numpy()
    moving = ants.from_numpy(moving)
    moving_mask = ants.from_numpy(moving_mask.astype(float))

    disp_tensor, warped_seg, reg_res = run_ants(fixed, moving, args.transform_type, f"{case_result_save_dir}/ants/", fixed_mask, moving_mask)

    # Compute the metrics
    flips_percentage_in_img, log_jac_std_in_img = compute_jacob_det_for_ants(reg_res['fwdtransforms'][0], fixed)
    results.append(
        (
            case_id,
            flips_percentage_in_img,
            log_jac_std_in_img
        )
    )

    disp_itk_format = (
        disp_tensor[0].double()
        .numpy()[list(reversed(range(3)))]
        .transpose([3,2,1,0])
    )
    
    # Save to output folders
    disp_itk_format = nib.Nifti1Image(disp_itk_format, affine=np.eye(4))
    nib.save(disp_itk_format, f"{footsteps.output_dir}/submission/NLST/disp_{fixed_path.split('_')[1]}_{moving_path.split('_')[1]}.nii.gz")

import subprocess

subprocess.call("zip -r submission.zip ./*", shell=True, cwd=f"{footsteps.output_dir}/submission/")  

df = pd.DataFrame(results, columns=[
    "case_id",
    "flips_percentage", "log_jac_std",])
df.to_csv(os.path.join(footsteps.output_dir, "results.csv"), index=False)

# Aggregate the metrics
aggregate = {
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