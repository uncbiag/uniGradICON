import argparse

import footsteps
import icon_registration.itk_wrapper as itk_wrapper
import itk
import numpy as np
import torch
import utils
import glob
import os
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import json

from unigradicon.train import make_network

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

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str, help="the path to the weights of the network")
parser.add_argument("--io_steps", type=int, default=0, help="Steps for IO")
parser.add_argument("--device", type=int, default=0, help="GPU ID.")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")
parser.add_argument("--bidirection", type=int, default=0, help="Whether to evaluate the registration in two directions.")
parser.add_argument("--ACDC_path", type=str, default="", help="the path to the folder containing ACDC dataset")
parser.add_argument("--MM_path", type=str, default="", help="the path to the folder containing MM dataset")

args = parser.parse_args()
weights_path = args.weights_path
device = torch.device(f'cuda:{args.device}')
torch.cuda.set_device(device)

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

# Create experiment folder
if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/ACDC")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

# Save the registration results
result_save_dir = os.path.join(footsteps.output_dir, "results")
os.makedirs(result_save_dir, exist_ok=True)


input_shape = [1, 1, 175, 175, 175]
net = make_network(input_shape, include_last_step=True)
logger.log(net.regis_net.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False))
net.eval()

shape_prod = np.prod(input_shape)

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
    
    case_id = case
    case_result_save_dir = os.path.join(result_save_dir, case_id)
    os.makedirs(case_result_save_dir, exist_ok=True)

    phi_AB, phi_BA, loss= itk_wrapper.register_pair(net, image_A, image_B, finetune_steps=None if args.io_steps == 0 else args.io_steps,
        return_artifacts=True,
    )

    interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_A)
    warped_segmentation_A = itk.resample_image_filter(
            segmentation_A, 
            transform=phi_AB,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=segmentation_B
            )
    
    itk.imwrite(segmentation_A, os.path.join(case_result_save_dir, f"{case_id}.ES_seg.nii.gz"))
    itk.imwrite(segmentation_B, os.path.join(case_result_save_dir, f"{case_id}.ED_seg.nii.gz"))
    itk.imwrite(warped_segmentation_A, os.path.join(case_result_save_dir, f"{case_id}.warped_ES_seg.nii.gz"))
    # itk.transformwrite(phi_AB, os.path.join(case_result_save_dir, f"{case_id}.trans_ES_ED.hdf5"))

    flips_percentage, log_jac_std = utils.compute_jacob_det_from_itk_transform(phi_AB, segmentation_B)
    flips_percentage_in_img, log_jac_std_in_img = utils.compute_jacob_det(net.phi_AB_vectorfield)

    metric = utils.compute_metrics(
        segmentation_B, segmentation_A, warped_segmentation_A, labels=utils.get_label_list(case_id))
    
    results.append(
        (
            f"{case_id}.ES_ED",
            os.path.join(case_result_save_dir, f"{case_id}.ES_seg.nii.gz"),
            os.path.join(case_result_save_dir, f"{case_id}.ED_seg.nii.gz"),
            os.path.join(case_result_save_dir, f"{case_id}.warped_ES_seg.nii.gz"),
            flips_percentage,
            log_jac_std,
            flips_percentage_in_img,
            log_jac_std_in_img,
            metric["dice"],
            metric["hd95"]
        )
    )

    if bidirection:
        # Swap the images
        phi_BA, phi_AB, loss= itk_wrapper.register_pair(net, image_B, image_A, finetune_steps=None if args.io_steps == 0 else args.io_steps,
            return_artifacts=True,
        )

        interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_B)
        warped_segmentation_B = itk.resample_image_filter(
                segmentation_B, 
                transform=phi_BA,
                interpolator=interpolator,
                use_reference_image=True,
                reference_image=segmentation_A
                )
        
        itk.imwrite(warped_segmentation_B, os.path.join(case_result_save_dir, f"{case_id}.warped_ED_seg.nii.gz"))
        # itk.transformwrite(phi_BA, os.path.join(case_result_save_dir, f"{case_id}.trans_ED_ES.hdf5"))
        flips_percentage, log_jac_std = utils.compute_jacob_det_from_itk_transform(phi_BA, segmentation_A)
        flips_percentage_in_img, log_jac_std_in_img = utils.compute_jacob_det(net.phi_BA_vectorfield)

        metric = utils.compute_metrics(
        segmentation_A, segmentation_B, warped_segmentation_B, labels=utils.get_label_list(case_id))

        results.append(
            (
                f"{case_id}.ED_ES",
                os.path.join(case_result_save_dir, f"{case_id}.ED_seg.nii.gz"),
                os.path.join(case_result_save_dir, f"{case_id}.ES_seg.nii.gz"),
                os.path.join(case_result_save_dir, f"{case_id}.warped_ED_seg.nii.gz"),
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