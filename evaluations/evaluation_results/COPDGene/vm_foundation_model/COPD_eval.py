import itk
import footsteps
import os

import torch
import utils
import numpy as np
import time

import icon_registration
import icon_registration.pretrained_models
import icon_registration.test_utils

import pandas as pd
import json


# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm 

image_root = "/playpen-raid1/lin.tian/data/lung/dirlab_highres_350"
landmark_root = "/playpen-raid1/lin.tian/data/lung/reg_lung_2d_3d_1000_dataset_4_proj_clean_bg/landmarks/"

cases = [f"copd{i}_highres" for i in range(1, 11)]

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
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/COPDGene")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

input_shape = [1, 1, 176, 176, 176]
net = vxm.networks.VxmDense.load(weights_path, device)
net.eval()

net.bidir = True

elapsed_times = []

result_details = {}
results = []

for case in cases:
    image_insp = itk.imread(f"{image_root}/{case}/{case}_INSP_STD_COPD_img.nii.gz")
    image_exp = itk.imread(f"{image_root}/{case}/{case}_EXP_STD_COPD_img.nii.gz")
    seg_insp = itk.imread(f"{image_root}/{case}/{case}_INSP_STD_COPD_label.nii.gz")
    seg_exp = itk.imread(f"{image_root}/{case}/{case}_EXP_STD_COPD_label.nii.gz")

    case_id = case

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
    phi_AB, phi_BA, flip, phi_AB_vectorfield, phi_BA_vectorfield = utils.register_pair(
        net,
        image_insp_preprocessed,
        image_exp_preprocessed,
        model_input_shape=input_shape,
        device=device
    )
    torch.cuda.synchronize()
    end = time.time()
    elapsed_times.append(end - start)

    TRE = {"target_pts":[], "moved_pts":[], "distances":[]}
    dists = []
    for i in range(len(landmarks_exp)):
        px, py = (
            landmarks_insp[i],
            np.array(phi_AB.TransformPoint(tuple(landmarks_exp[i]))),
        )
        dists.append(np.sqrt(np.sum((px - py) ** 2)))
        TRE["target_pts"].append(px)
        TRE["moved_pts"].append(py)
        TRE["distances"].append(dists[-1])
    
    # Compute the metrics
    flips_percentage, log_jac_std = utils.compute_jacob_det_from_itk_transform(phi_AB, image_exp_preprocessed)
    flips_percentage_in_img, log_jac_std_in_img = utils.compute_jacob_det(phi_AB_vectorfield)

    result_details[f"COPDGene.{case_id}.insp_to_exp"] = {"TRE": TRE}
    results.append(
        (
            f"COPDGene.{case_id}.insp_to_exp",
            flips_percentage,
            log_jac_std,
            flips_percentage_in_img,
            log_jac_std_in_img,
            np.mean(dists),
        )
    )
    
    # We report the mTRE of this direction.
    # Because this is what is reported in pTVreg (the SOTA conventional method)
    TRE = {"target_pts":[], "moved_pts":[], "distances":[]}
    dists = []
    for i in range(len(landmarks_insp)):
        px, py = (
            landmarks_exp[i],
            np.array(phi_BA.TransformPoint(tuple(landmarks_insp[i]))),
        )
        dists.append(np.sqrt(np.sum((px - py) ** 2)))
        TRE["target_pts"].append(px)
        TRE["moved_pts"].append(py)
        TRE["distances"].append(dists[-1])
    
    # Compute the metrics
    flips_percentage, log_jac_std = utils.compute_jacob_det_from_itk_transform(phi_BA, image_insp_preprocessed)
    flips_percentage_in_img, log_jac_std_in_img = utils.compute_jacob_det(phi_BA_vectorfield)

    result_details[f"COPDGene.{case_id}.exp_to_insp"] = {"TRE": TRE}
    results.append(
        (
            f"COPDGene.{case_id}.exp_to_insp",
            flips_percentage,
            log_jac_std,
            flips_percentage_in_img,
            log_jac_std_in_img,
            np.mean(dists),
        )
    )


df = pd.DataFrame(results, columns=[
    "case_id",
    "flips_percentage_itk", "log_jac_std_itk",
    "flips_percentage", "log_jac_std",
    "mTRE"])
df.to_csv(os.path.join(footsteps.output_dir, "results.csv"), index=False)

# Aggregate the metrics
df_insp_to_exp = df[df["case_id"].str.contains("insp_to_exp")]
df_exp_to_insp = df[df["case_id"].str.contains("exp_to_insp")]
aggregate = {
    "insp_to_exp":{
        "mTRE": {"mean": np.mean(df_insp_to_exp["mTRE"]), "std": np.std(df_insp_to_exp["mTRE"])},
        "flips_percentage_itk": {"mean": np.mean(df_insp_to_exp["flips_percentage_itk"]), "std": np.std(df_insp_to_exp["flips_percentage_itk"])},
        "log_jac_std_itk": {"mean": np.mean(df_insp_to_exp["log_jac_std_itk"]), "std": np.std(df_insp_to_exp["log_jac_std_itk"])},
        "flips_percentage": {"mean": np.mean(df_insp_to_exp["flips_percentage"]), "std": np.std(df_insp_to_exp["flips_percentage"])},
        "log_jac_std": {"mean": np.mean(df_insp_to_exp["log_jac_std"]), "std": np.std(df_insp_to_exp["log_jac_std"])}
    },
    "exp_to_insp":{
        "mTRE": {"mean": np.mean(df_exp_to_insp["mTRE"]), "std": np.std(df_exp_to_insp["mTRE"])},
        "flips_percentage_itk": {"mean": np.mean(df_exp_to_insp["flips_percentage_itk"]), "std": np.std(df_exp_to_insp["flips_percentage_itk"])},
        "log_jac_std_itk": {"mean": np.mean(df_exp_to_insp["log_jac_std_itk"]), "std": np.std(df_exp_to_insp["log_jac_std_itk"])},
        "flips_percentage": {"mean": np.mean(df_exp_to_insp["flips_percentage"]), "std": np.std(df_exp_to_insp["flips_percentage"])},
        "log_jac_std": {"mean": np.mean(df_exp_to_insp["log_jac_std"]), "std": np.std(df_exp_to_insp["log_jac_std"])}
    }
}

logger.log("Aggregate:")
for k, v in aggregate.items():
    logger.log(f"{k}:")
    for k2, v2 in v.items():
        logger.log(f"  {k2}: {v2['mean']:.2f} ± {v2['std']:.2f}")

def convert_numpy_float(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return json.JSONEncoder.default(obj)

with open(os.path.join(footsteps.output_dir, "aggregate_results.json"), 'w') as f:
    json.dump(aggregate, f, default=convert_numpy_float, indent=4)

# Save the detailed results
result_details_all = {"cases": result_details, "aggregates": aggregate}
with open(os.path.join(footsteps.output_dir, "detailed_results.json"), 'w') as f:
    json.dump(result_details_all, f, default=convert_numpy_float, indent=4)

logger.close()
