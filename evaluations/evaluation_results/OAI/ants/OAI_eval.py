import footsteps
import itk
import ants
import numpy as np
import argparse
import os
import pandas as pd
import json
from tqdm import tqdm
from ants_helper import compute_jacob_det_for_ants, copy_reference_image_info, run_ants

import sys
# Create the absolute path by joining the current directory and the relative path
absolute_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(absolute_path)
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, default="", help="Experiment name.")
parser.add_argument("--transform_type", type=str, default="SyN", help="Transform Type: SyN or SyNOnly")

args = parser.parse_args()
if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/{args.transform_type}/OAI")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

# Save the registration results
result_save_dir = os.path.join(footsteps.output_dir, "results")
os.makedirs(result_save_dir, exist_ok=True)

with open("/playpen-raid2/lin.tian/projects/icon_lung/ICON/training_scripts/oai_paper_pipeline/splits/test/pair_path_list.txt") as f:
    test_pair_paths = f.readlines()

results = []
for test_pair_path in tqdm(test_pair_paths):
    test_pair_path = test_pair_path.replace("playpen", "playpen-raid").split()

    case_id_A = test_pair_path[0].split("/")[-1].split(".")[0]
    case_id_B = test_pair_path[1].split("/")[-1].split(".")[0]
    case_id = f"OAI.{case_id_A}_to_{case_id_B}"
    case_result_save_dir = os.path.join(result_save_dir, case_id)
    os.makedirs(case_result_save_dir, exist_ok=True)
    os.makedirs(os.path.join(case_result_save_dir, "ants"), exist_ok=True)

    test_pair = [itk.imread(path) for path in test_pair_path]
    test_pair = [
            (itk.flip_image_filter(t, flip_axes=(False, False, True))
                if "RIGHT" in path else t 
                ) for (t , path) in zip(test_pair, test_pair_path)]
    
    image_A, image_B, segmentation_A, segmentation_B = test_pair
    test_pair = [ants.from_numpy(itk.array_from_image(t)) for t in test_pair]

    moving, fixed, moving_seg, fixed_seg = test_pair

    _, warped_seg, reg_res = run_ants(fixed, moving, args.transform_type, f"{case_result_save_dir}/ants/", fixed_mask=fixed_seg, moving_mask=moving_seg, return_disp_tensor=False)

    warped_segmentation_A = copy_reference_image_info(np.ascontiguousarray(warped_seg.numpy()), segmentation_B)
    # Save the registration results
    itk.imwrite(segmentation_A, os.path.join(case_result_save_dir, f"{case_id}.{case_id_A}_seg.nii.gz"))
    itk.imwrite(segmentation_B, os.path.join(case_result_save_dir, f"{case_id}.{case_id_B}_seg.nii.gz"))
    itk.imwrite(warped_segmentation_A, os.path.join(case_result_save_dir, f"{case_id}.warped_{case_id_A}_seg.nii.gz"))

    flips_percentage_in_img, log_jac_std_in_img = compute_jacob_det_for_ants(reg_res['fwdtransforms'][0], fixed)

    metric = utils.compute_metrics(
        segmentation_B, segmentation_A, warped_segmentation_A, labels=utils.get_label_list(case_id))
    
    results.append(
        (
            case_id,
            os.path.join(case_result_save_dir, f"{case_id}.{case_id_A}_seg.nii.gz"),
            os.path.join(case_result_save_dir, f"{case_id}.{case_id_B}_seg.nii.gz"),
            os.path.join(case_result_save_dir, f"{case_id}.warped_{case_id_A}_seg.nii.gz"),
            flips_percentage_in_img,
            log_jac_std_in_img,
            metric["dice"],
            metric["hd95"]
        )
    )

df = pd.DataFrame(results, columns=[
    "case_id", "segmentation_A", 
    "segmentation_B", "warped_segmentation_A",
    "flips_percentage", "log_jac_std",
    "dice", "hd95"])
df.to_csv(os.path.join(footsteps.output_dir, "results.csv"), index=False)

# Aggregate the metrics
dices = [dice*100. for dice in df["dice"].tolist()]
aggregate = {
    "dice": {"mean": np.mean(dices), "std": np.std(dices)},
    "hd95": {"mean": np.mean(df["hd95"]), "std": np.std(df["hd95"])},
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