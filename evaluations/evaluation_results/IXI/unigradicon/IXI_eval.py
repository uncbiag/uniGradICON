import argparse
import glob

import footsteps
import icon_registration as icon
import icon_registration.itk_wrapper as itk_wrapper
import itk
import numpy as np
import torch
import utils
from IXI_utils import (Compose, IXIBrainInferDataset, NumpyType, Seg_norm,
                       get_label_id)
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import os
import pandas as pd
import json
from tqdm import tqdm

from unigradicon.train import make_network

origin_shape = [1, 1, 160, 192, 224]
input_shape = [1, 1, 175, 175, 175]

def preprocess(image):
    # image = itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.F, 3]].New()(image)
    max_ = float(np.max(np.array(image)))
    image = itk.shift_scale_image_filter(image, shift=0., scale = .9 / max_)
    
    #image = itk.clamp_image_filter(image, bounds=(0, 1))
    return image

def array_to_itk_image(array):
    itk_image = itk.GetImageFromArray(array)
    itk_image.SetOrigin([0.0, 0.0, 0.0])  # Set origin
    itk_image.SetSpacing([1.0, 1.0, 1.0]) # Set spacing
    itk_image.SetDirection(np.identity(3)) # Set direction
    return itk_image

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str, help="the path to the weights of the network")
parser.add_argument("--io_steps", type=int, default=0, help="Steps for IO")
parser.add_argument("--device", type=int, default=0, help="GPU ID.")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")
parser.add_argument("--data_folder", type=str, default="", help="Path to the dataset directory.")

args = parser.parse_args()
weights_path = args.weights_path
device = torch.device(f'cuda:{args.device}')
torch.cuda.set_device(device)

if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/IXI")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

# Save the registration results
result_save_dir = os.path.join(footsteps.output_dir, "results")
os.makedirs(result_save_dir, exist_ok=True)

net = make_network(input_shape, include_last_step=True)#, framework="svf")
logger.log(net.regis_net.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False))
net.to(device)
net.eval()


atlas_dir = f'{args.data_folder}/atlas.pkl'
test_dir = f'{args.data_folder}/Test/'
test_composed = Compose([Seg_norm(),
                        NumpyType((np.float32, np.int16)),
                        ])
test_set = IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

spacing = 1.0 / (np.array(origin_shape[2::]) - 1)
identity = torch.from_numpy(icon.mermaidlite.identity_map_multiN(origin_shape, spacing)).to(device)

print(f"IXI labels:{get_label_id(f'{args.data_folder}/label_info.txt')}")
print(f"The labels we use for DICE: {utils.get_label_list('IXI.1')}")

results = []
origin_state_dict = copy.deepcopy(net.state_dict())
for moving, fixed, moving_seg, fixed_seg, path in tqdm(test_loader):
    net.load_state_dict(origin_state_dict)

    image_A = preprocess(array_to_itk_image(moving.numpy()[0,0]))
    image_B = preprocess(array_to_itk_image(fixed.numpy()[0,0]))
    segmentation_A = array_to_itk_image(moving_seg.cpu().numpy()[0,0])
    segmentation_B = array_to_itk_image(fixed_seg.cpu().numpy()[0,0])

    case_id = f'IXI.{path[0].split("/")[-1].split(".")[0].split("_")[1]}'
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
    
    # Save the registration results
    itk.imwrite(segmentation_A, os.path.join(case_result_save_dir, f"{case_id}.atlas_seg.nii.gz"))
    itk.imwrite(segmentation_B, os.path.join(case_result_save_dir, f"{case_id}.{case_id.split('.')[1]}_seg.nii.gz"))
    itk.imwrite(warped_segmentation_A, os.path.join(case_result_save_dir, f"{case_id}.warped_atlas_seg.nii.gz"))

    # Compute the metrics
    flips_percentage, log_jac_std = utils.compute_jacob_det_from_itk_transform(phi_AB, segmentation_B)
    flips_percentage_in_img, log_jac_std_in_img = utils.compute_jacob_det(net.phi_AB_vectorfield)

    metric = utils.compute_metrics(
        segmentation_B, segmentation_A, warped_segmentation_A, labels=utils.get_label_list(case_id))

    results.append(
        (
            case_id,
            os.path.join(case_result_save_dir, f"{case_id}.atlas_seg.nii.gz"),
            os.path.join(case_result_save_dir, f"{case_id}.{case_id.split('.')[1]}_seg.nii.gz"),
            os.path.join(case_result_save_dir, f"{case_id}.warped_atlas_seg.nii.gz"),
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

