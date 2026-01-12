import footsteps
import icon_registration.itk_wrapper
import icon_registration.test_utils
import itk
import numpy as np
import torch
import utils
import os
import pandas as pd
import json
from tqdm import tqdm

from unigradicon import make_network

image_root = "/playpen-raid1/lin.tian/data/lung/dirlab_highres_350"
landmark_root = "/playpen-raid1/lin.tian/data/lung/reg_lung_2d_3d_1000_dataset_4_proj_clean_bg/landmarks/"

cases = [f"copd{i}_highres" for i in range(1, 11)]

def preprocess(image: "itk.Image",
                            segmentation: "itk.Image") -> "itk.Image":

    image = itk.clamp_image_filter(image, Bounds=(-1000, 1000))
    cast_filter = itk.CastImageFilter[type(image), itk.Image.F3].New()
    cast_filter.SetInput(image)
    cast_filter.Update()
    image = cast_filter.GetOutput()

    segmentation_cast_filter = itk.CastImageFilter[type(segmentation),
                                                   itk.Image.F3].New()
    segmentation_cast_filter.SetInput(segmentation)
    segmentation_cast_filter.Update()
    segmentation = segmentation_cast_filter.GetOutput()

    image = itk.shift_scale_image_filter(image, shift=1000, scale=1 / 2000)

    mask_filter = itk.MultiplyImageFilter[itk.Image.F3, itk.Image.F3,
                                          itk.Image.F3].New()

    mask_filter.SetInput1(image)
    mask_filter.SetInput2(segmentation)
    mask_filter.Update()

    return mask_filter.GetOutput()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str, help="the path to the weights of the network")
parser.add_argument("--io_steps", type=int, default=0, help="Steps for IO")
parser.add_argument("--device", type=int, default=0, help="GPU ID.")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")

args = parser.parse_args()
weights_path = args.weights_path
device = torch.device(f'cuda:{args.device}')
torch.cuda.set_device(device)

if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/COPDGene")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

input_shape = [1, 1, 175, 175, 175]
net = make_network(input_shape, include_last_step=True)
logger.log(net.regis_net.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False))
net.eval()


result_details = {}
results = []

for case in tqdm(cases):
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
        preprocess(
            image_insp, seg_insp
        )
    )
    image_exp_preprocessed = (
        preprocess(image_exp, seg_exp)
    )

    phi_AB, phi_BA, loss = icon_registration.itk_wrapper.register_pair(
        net,
        image_insp_preprocessed,
        image_exp_preprocessed,
        finetune_steps=None if args.io_steps == 0 else args.io_steps,
        return_artifacts=True,
    )
    
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
    flips_percentage_in_img, log_jac_std_in_img = utils.compute_jacob_det(net.phi_AB_vectorfield)

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
    flips_percentage_in_img, log_jac_std_in_img = utils.compute_jacob_det(net.phi_BA_vectorfield)

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