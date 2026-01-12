import argparse
import json
import os

import footsteps
import itk
import numpy as np
import torch
from scipy.ndimage.interpolation import zoom as zoom
from tqdm import tqdm
import ants
import pandas as pd
import json

from ants_helper import compute_jacob_det_for_ants, run_ants

import sys
# Create the absolute path by joining the current directory and the relative path
absolute_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(absolute_path)
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, help="the path to the folder containing learn2reg AbdomenMRCT dataset")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")
parser.add_argument("--transform_type", type=str, default="SyN", help="Transform Type: SyN or SyNOnly")

clamp = [-1000, 1000]

args = parser.parse_args()

args = parser.parse_args()
if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/{args.transform_type}/L2R_abdomenMRCT")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")
# Save the registration results
result_save_dir = os.path.join(footsteps.output_dir, "results")
os.makedirs(result_save_dir, exist_ok=True)

os.makedirs(f"{footsteps.output_dir}/submission/submission/task_01", exist_ok=True)


def preprocess(img):
    im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
    img = torch.clip(img, im_min, im_max)
    img = (img-im_min) / (im_max-im_min)
    return img

# L2R changed the idx of the image but kept the idx in the evaluation script the same as the old id.
cases_dict = {"0006":"0012", "0007":"0014", "0008":"0016"}

with open(f"{args.data_folder}/AbdomenMRCT_dataset.json", 'r') as data_info:
    data_info = json.loads(data_info.read())
test_cases = [[c["fixed"], c["moving"]] for c in data_info["registration_val"]]

results = []
for (fixed_path, moving_path) in tqdm(test_cases):
    case_id = cases_dict[fixed_path.split('_')[1]]
    case_result_save_dir = os.path.join(result_save_dir, case_id)
    os.makedirs(case_result_save_dir, exist_ok=True)
    os.makedirs(os.path.join(case_result_save_dir, "ants"), exist_ok=True)

    fixed = np.asarray(itk.imread(os.path.join(args.data_folder, fixed_path)))
    moving = np.asarray(itk.imread(os.path.join(args.data_folder, moving_path)))

    # Fixed: MRI
    fixed = torch.Tensor(np.array(fixed))
    fixed = preprocess(fixed)
    fixed = ants.from_numpy(fixed.numpy())
    
    # Move: CT
    moving = torch.Tensor(np.array(moving))
    moving = (torch.clamp(moving, clamp[0], clamp[1]) - clamp[0])/(clamp[1]-clamp[0])
    moving = ants.from_numpy(moving.numpy())

    disp_tensor, _, reg_res = run_ants(fixed, moving, args.transform_type, f"{case_result_save_dir}/ants/")

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
        disp_tensor[0].float()
        .numpy()[list(reversed(range(3)))]
        .transpose([0,3,2,1])
    )


    # Save to output folders in the format required by l2r evaluation script
    disp_x = zoom(disp_itk_format[0], 0.5, order=2).astype('float16')
    disp_y = zoom(disp_itk_format[1], 0.5, order=2).astype('float16')
    disp_z = zoom(disp_itk_format[2], 0.5, order=2).astype('float16')
    disp = np.array((disp_x, disp_y, disp_z))
    case_id = cases_dict[fixed_path.split('_')[1]]
    np.savez_compressed(f"{footsteps.output_dir}/submission/submission/task_01/disp_{case_id}_{case_id}.npz", disp)

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