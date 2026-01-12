import argparse
import os

import ants
import footsteps
import numpy as np
import torch
from scipy.ndimage.interpolation import zoom as zoom
from tqdm import tqdm
import pandas as pd
import json

from ants_helper import compute_jacob_det_for_ants, run_ants

import sys
# Create the absolute path by joining the current directory and the relative path
absolute_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(absolute_path)
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, help="the path to the folder containing learn2reg AbdomenCTCT dataset")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")
parser.add_argument("--transform_type", type=str, default="SyN", help="Transform Type: SyN or SyNOnly")

origin_shape = [1, 1, 160, 192, 224] # The submission system asks for 80 x 96 x 112
input_shape = [1, 1, 175, 175, 175]

args = parser.parse_args()
if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/{args.transform_type}/L2R_OASIS")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")
# Save the registration results
result_save_dir = os.path.join(footsteps.output_dir, "results")
os.makedirs(result_save_dir, exist_ok=True)

os.makedirs(f"{footsteps.output_dir}/submission/submission/task_03", exist_ok=True)



import glob

cases = sorted(glob.glob(f"{args.data_folder}/val/img????.nii.gz"))
test_cases = []
for i in range(len(cases[:-1])):
    test_cases.append([cases[i], cases[i+1]])


results = []
for (fixed_path, moving_path) in tqdm(test_cases):
    case_id = f"{fixed_path.split('/')[-1][3:7]}_{moving_path.split('/')[-1][3:7]}"
    case_result_save_dir = os.path.join(result_save_dir, case_id)
    os.makedirs(case_result_save_dir, exist_ok=True)
    os.makedirs(os.path.join(case_result_save_dir, "ants"), exist_ok=True)

    fixed = ants.image_read(os.path.join(args.data_folder, fixed_path))
    moving = ants.image_read(os.path.join(args.data_folder, moving_path))
    fixed_mask = ants.image_read(os.path.join(args.data_folder, fixed_path.replace("img", "seg")))
    moving_mask = ants.image_read(os.path.join(args.data_folder, moving_path.replace("img", "seg")))

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

    disp_tensor = disp_tensor[0].numpy()

    # Save to output folders in the format required by l2r evaluation script
    disp_x = zoom(disp_tensor[0], 0.5, order=2).astype('float16')
    disp_y = zoom(disp_tensor[1], 0.5, order=2).astype('float16')
    disp_z = zoom(disp_tensor[2], 0.5, order=2).astype('float16')
    disp = np.array((disp_x, disp_y, disp_z))
    np.savez_compressed(f"{footsteps.output_dir}/submission/submission/task_03/disp_{fixed_path.split('/')[-1][3:7]}_{moving_path.split('/')[-1][3:7]}.npz", disp)

with open(f"{footsteps.output_dir}/submission/submission/test.txt", "w") as f:
    f.write("dummy file.")

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