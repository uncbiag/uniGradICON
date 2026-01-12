import argparse

import footsteps
import itk
import numpy as np
import torch
import glob
import os
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import json
import ants
from ants_helper import compute_jacob_det_for_ants, copy_reference_image_info, run_ants

import sys
# Create the absolute path by joining the current directory and the relative path
absolute_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(absolute_path)
import utils


segmentation_labels = {
    "ACDC":{
        1:  "RV", # Right Ventricle
        2:  "MYO", # Myocardium
        3:  "LV", # Left Ventricle
    },
    "MM":{
        1:  "LV", # Left Ventricle
        2:  "MYO", # Myocardium
        3:  "RV", # Right Ventricle
    }
}

def quantile(arr: torch.Tensor, q):
    arr = arr.flatten()
    l = len(arr)
    return torch.kthvalue(arr, int(q * l)).values

def preprocess(image):
    image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
    min_, _ = itk.image_intensity_min_max(image)
    max_ = quantile(torch.tensor(np.array(image)), .99).item()
    image = itk.clamp_image_filter(image, Bounds=(min_, max_))
    image = itk.shift_scale_image_filter(image, shift=-min_, scale = 1/(max_-min_)) 
    return image

def to_itk(img, affine):
    img_itk = itk.GetImageFromArray(img)
    
    # Set the origin, spacing and direction in the ITK image
    img_itk.SetOrigin(affine[:3, 3])
    img_itk.SetSpacing(np.linalg.norm(affine[:3, :3], axis=0))
    img_itk.SetDirection(affine[:3, :3]/np.linalg.norm(affine[:3, :3], axis=0)[:, None])
    return img_itk

def permute(image):
    return itk.permute_axes_image_filter(image, order=[2, 1, 0])

def slice_and_permute(image, slice_id, type=itk.F):
    filter = itk.ExtractImageFilter[itk.Image[type,4], itk.Image[type,3]].New(image)
    region = itk.ImageRegion[4]()
    region.SetIndex([0, 0, 0, slice_id])
    shape = image.shape
    region.SetSize([shape[3], shape[2], shape[1], 0])
    filter.SetExtractionRegion(region)
    filter.SetDirectionCollapseToIdentity()
    filter.Update()
    new_img = filter.GetOutput()
    image = itk.permute_axes_image_filter(new_img, order=[2, 1, 0])
    return image

def compute_jacob_det(phi, mask=None):
    '''
    Args:
        transformation: 1x3xHxWxD (in GPU)
        mask: 1x1xHxWxD
    '''
    a = (phi[:, :, 1:, 1:, 1:] - phi[:, :, :-1, 1:, 1:]).detach()
    b = (phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, :-1, 1:]).detach()
    c = (phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, 1:, :-1]).detach()

    dV = torch.sum(torch.cross(a, b, 1) * c, axis=1)[0]
    jacob_np = dV.cpu().numpy()
    if mask is not None:
        mask = np.array(mask)[0,0, :-1, :-1, :-1]
        mask[mask>0] = 1
        jacob_np = np.ma.MaskedArray(jacob_np, mask)
    flips_percentage = np.mean(jacob_np<0) * 100.

    # Following the implementation in Learn2Reg
    # https://github.com/MDL-UzL/L2R/blob/main/evaluation/evaluation.py#L139
    log_jac_det_std = np.log((jacob_np+3).clip(1e-9, 1e9)).std() 

    return flips_percentage, log_jac_det_std

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, default="", help="Experiment name.")
parser.add_argument("--bidirection", type=int, default=0, help="Whether to evaluate the registration in two directions.")
parser.add_argument("--ACDC_path", type=str, default="", help="the path to the folder containing ACDC dataset")
parser.add_argument("--MM_path", type=str, default="", help="the path to the folder containing MM dataset")
parser.add_argument("--transform_type", type=str, default="SyN", help="Transform Type: SyN or SyNOnly")

args = parser.parse_args()

bidirection = args.bidirection

# Check if the ACDC_path and MM_path are provided
if args.ACDC_path == "" and args.MM_path == "":
    raise ValueError("Please provide the paths to the ACDC or MM datasets using --ACDC_path and --MM_path arguments.")

if args.ACDC_path != "":
    ACDC_cases = sorted(glob.glob(f'{args.ACDC_path}/database/testing/patient*/'))
else:
    ACDC_cases = []

if args.MM_path != "":
    MM_cases = sorted(glob.glob(f'{args.MM_path}/Validation/*/*_sa.nii.gz'))
    df = pd.read_csv(f"{args.MM_path}/211230_MMs_Dataset_information_diagnosis_opendataset.csv")
else:
    MM_cases = []
    df = None

# Create the experiment directory
if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/{args.transform_type}/ACDCMM")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")
os.makedirs(f"{footsteps.output_dir}/tmp", exist_ok=True)

# Save the registration results
result_save_dir = os.path.join(footsteps.output_dir, "results")
os.makedirs(result_save_dir, exist_ok=True)

# Generate the data pairs
test_cases = {}
for case in ACDC_cases:
    case_pathes = {}
    case_id = case.split("/")[-2]
    assert case_id not in test_cases, f"Case {case_id} already exists in test_cases."

    with open(os.path.join(case, "Info.cfg")) as f:
        for line in f.readlines():
            if "ES" in line:
                ES_id = int(line.split(":")[-1][:-1])
                case_pathes["ES"] = os.path.join(case, f"{case_id}_frame{ES_id:02d}.nii.gz")
                case_pathes["ES_seg"] = os.path.join(case, f"{case_id}_frame{ES_id:02d}_gt.nii.gz")
                
            elif "ED" in line:
                ED_id = int(line.split(":")[-1][:-1])
                case_pathes["ED"] = os.path.join(case, f"{case_id}_frame{ED_id:02d}.nii.gz")
                case_pathes["ED_seg"] = os.path.join(case, f"{case_id}_frame{ED_id:02d}_gt.nii.gz")
    
    test_cases[f"ACDC.{case_id}"] = case_pathes

for case in MM_cases:
    case_pathes = {}
    case_id = case.split("/")[-1].split("_")[0]

    ES_slice_id = df[df["External code"] == case_id]["ES"].values[0]
    ED_slice_id = df[df["External code"] == case_id]["ED"].values[0]
    case_pathes["ES"] = int(ES_slice_id)
    case_pathes["ED"] = int(ED_slice_id)
    case_pathes["img"] = case
    case_pathes["seg"] = case.split(".")[0] + "_gt.nii.gz"
    test_cases[f"MM.{case_id}"] = case_pathes

print(f"Evaluating {len(test_cases)} cases")
results = []

for case in tqdm(test_cases):
    paired_pathes = []
    if "ACDC" in case:
        image_A = nib.load(test_cases[case]["ES"])
        image_A = preprocess(permute(to_itk(image_A.get_fdata(), image_A.affine)))
        image_B = nib.load(test_cases[case]["ED"])
        image_B = preprocess(permute(to_itk(image_B.get_fdata(), image_B.affine)))
        segmentation_A = nib.load(test_cases[case]["ES_seg"])
        segmentation_A = permute(to_itk(segmentation_A.get_fdata(), segmentation_A.affine))
        segmentation_B = nib.load(test_cases[case]["ED_seg"])
        segmentation_B = permute(to_itk(segmentation_B.get_fdata(), segmentation_B.affine))

        # print(f"ACDC {case} image shape: {np.array(image_A).shape}, {np.array(image_B).shape}, {np.array(segmentation_A).shape}, {np.array(segmentation_B).shape}")

    else:
        image_nib = nib.load(test_cases[case]["img"])
        image = image_nib.get_fdata()
        image_A = preprocess(permute(to_itk(image[:,:,:,test_cases[case]["ES"]], image_nib.affine)))
        image_B = preprocess(permute(to_itk(image[:,:,:,test_cases[case]["ED"]], image_nib.affine)))
        segmentation_nib = nib.load(test_cases[case]["seg"])
        segmentation = segmentation_nib.get_fdata()
        segmentation_A = permute(to_itk(segmentation[:,:,:,test_cases[case]["ES"]], segmentation_nib.affine))
        segmentation_B = permute(to_itk(segmentation[:,:,:,test_cases[case]["ED"]], segmentation_nib.affine))
        # print(f"MM {case} image shape: {np.array(image_A).shape}, {np.array(image_B).shape}, {np.array(segmentation_A).shape}, {np.array(segmentation_B).shape}")
    
    # Convert to Ants format
    fixed = ants.from_numpy(np.array(image_B))
    moving = ants.from_numpy(np.array(image_A))
    fixed_seg = ants.from_numpy(np.array(segmentation_B))
    moving_seg = ants.from_numpy(np.array(segmentation_A))
    
    case_id = case
    case_result_save_dir = os.path.join(result_save_dir, case_id)
    os.makedirs(case_result_save_dir, exist_ok=True)

    # Run ants registration
    disp_tensor, warped_seg, reg_res = run_ants(fixed, moving, args.transform_type, f"{footsteps.output_dir}/tmp/",
                                             fixed_mask=fixed_seg, moving_mask=moving_seg, return_disp_tensor=True)

    # disp_itk = itk.imread(reg_res['fwdtransforms'][0])

    # Convert the warped segmentation to itk image
    # The convertion between numpy and itk is always confusing.
    # Transpose the segmentation map to prevent the incorrect reversion of the order of the axes
    warped_segmentation_A = copy_reference_image_info(np.ascontiguousarray(warped_seg.numpy()), image_B)
    itk.imwrite(segmentation_A, os.path.join(case_result_save_dir, f"{case_id}.ES_seg.nii.gz"))
    itk.imwrite(segmentation_B, os.path.join(case_result_save_dir, f"{case_id}.ED_seg.nii.gz"))
    itk.imwrite(warped_segmentation_A, os.path.join(case_result_save_dir, f"{case_id}.warped_ES_seg.nii.gz"))

    flips_percentage_in_img, log_jac_std_in_img = compute_jacob_det_for_ants(reg_res['fwdtransforms'][0], fixed)

    metric = utils.compute_metrics(
        segmentation_B, segmentation_A, warped_segmentation_A, labels=utils.get_label_list(case_id))
    
    results.append(
        (
            f"{case_id}.ES_ED",
            os.path.join(case_result_save_dir, f"{case_id}.ES_seg.nii.gz"),
            os.path.join(case_result_save_dir, f"{case_id}.ED_seg.nii.gz"),
            os.path.join(case_result_save_dir, f"{case_id}.warped_ES_seg.nii.gz"),
            flips_percentage_in_img,
            log_jac_std_in_img,
            metric["dice"],
            metric["hd95"]
        )
    )

    if bidirection:
        # Swap the images
        fixed = ants.from_numpy(np.array(image_A))
        moving = ants.from_numpy(np.array(image_B))
        fixed_seg = ants.from_numpy(np.array(segmentation_A))
        moving_seg = ants.from_numpy(np.array(segmentation_B))

        # Run ants registration
        disp_tensor, warped_seg, reg_res = run_ants(fixed, moving, args.transform_type, f"{footsteps.output_dir}/tmp/",
                                                fixed_mask=fixed_seg, moving_mask=moving_seg, return_disp_tensor=True)

        # Convert the warped segmentation to itk image
        warped_segmentation_B = copy_reference_image_info(np.ascontiguousarray(warped_seg.numpy()), image_A)
        itk.imwrite(warped_segmentation_B, os.path.join(case_result_save_dir, f"{case_id}.warped_ED_seg.nii.gz"))

        flips_percentage_in_img, log_jac_std_in_img = compute_jacob_det_for_ants(reg_res['fwdtransforms'][0], fixed)
        
        metric = utils.compute_metrics(
        segmentation_A, segmentation_B, warped_segmentation_B, labels=utils.get_label_list(case_id))

        results.append(
            (
                f"{case_id}.ED_ES",
                os.path.join(case_result_save_dir, f"{case_id}.ED_seg.nii.gz"),
                os.path.join(case_result_save_dir, f"{case_id}.ES_seg.nii.gz"),
                os.path.join(case_result_save_dir, f"{case_id}.warped_ED_seg.nii.gz"),
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