import argparse
import json
import os

import footsteps
import icon_registration as icon
import itk
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import utils
from scipy.ndimage.interpolation import zoom as zoom
from tqdm import tqdm
import copy
import pandas as pd

from unigradicon import make_network
from utils import finetune_execute

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str, help="the path to the weights of the network")
parser.add_argument("--data_folder", type=str, help="the path to the folder containing learn2reg AbdomenCTCT dataset")
parser.add_argument("--io_steps", type=int, default=0, help="Steps for IO")
parser.add_argument("--device", type=int, default=0, help="GPU ID.")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")


origin_shape = [1, 1, 224, 192, 224]
input_shape = [1, 1, 175, 175, 175]
clamp = [-1000, 1000]

args = parser.parse_args()
weights_path = args.weights_path
device = torch.device(f'cuda:{args.device}')
torch.cuda.set_device(device)

if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/L2R_NLST")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

os.makedirs(f"{footsteps.output_dir}/submission/NLST", exist_ok=True)



net = make_network(input_shape, include_last_step=True)

logger.log(net.regis_net.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True))
net.to(device)
net.eval()


with open(f"{args.data_folder}/NLST_dataset.json", 'r') as data_info:
    data_info = json.loads(data_info.read())
test_cases = [[c["fixed"], c["moving"]] for c in data_info["registration_val"]]

spacing = 1.0 / (np.array(origin_shape[2::]) - 1)
identity = torch.from_numpy(icon.mermaidlite.identity_map_multiN(origin_shape, spacing)).to(device)

results = []
original_state_dict = copy.deepcopy(net.state_dict())
for (fixed_path, moving_path) in tqdm(test_cases):
    # Restore net weight in case we ran IO
    net.load_state_dict(original_state_dict)

    case_id = f"{fixed_path.split('_')[1]}_{moving_path.split('_')[1]}"

    fixed = np.asarray(itk.imread(os.path.join(args.data_folder, fixed_path)))
    moving = np.asarray(itk.imread(os.path.join(args.data_folder, moving_path)))
    fixed_mask = np.asarray(itk.imread(os.path.join(args.data_folder, fixed_path.replace("imagesTr", "masksTr"))))
    moving_mask = np.asarray(itk.imread(os.path.join(args.data_folder, moving_path.replace("imagesTr", "masksTr"))))

    fixed = torch.Tensor(np.array(fixed)).unsqueeze(0).unsqueeze(0)
    fixed = (torch.clamp(fixed, clamp[0], clamp[1]) - clamp[0])/(clamp[1]-clamp[0])
    fixed = fixed * torch.Tensor(fixed_mask).unsqueeze(0).unsqueeze(0)
    fixed_in_net = F.interpolate(fixed, input_shape[2:], mode='trilinear', align_corners=False)
    
    moving = torch.Tensor(np.array(moving)).unsqueeze(0).unsqueeze(0)
    moving = (torch.clamp(moving, clamp[0], clamp[1]) - clamp[0])/(clamp[1]-clamp[0])
    moving = moving * torch.Tensor(moving_mask).unsqueeze(0).unsqueeze(0)
    moving_in_net = F.interpolate(moving, input_shape[2:], mode='trilinear', align_corners=False)

    if args.io_steps > 0:
        loss = finetune_execute(net, moving_in_net.to(device), fixed_in_net.to(device), args.io_steps)

    with torch.no_grad():
        net(moving_in_net.to(device), fixed_in_net.to(device))

        # phi_AB and phi_BA are [1, 3, H, W, D] pytorch tensors representing the forward and backward
        # maps computed by the model
        phi_AB = net.phi_AB(identity)

        # Compute the metrics
        flips_percentage_in_img, log_jac_std_in_img = utils.compute_jacob_det(net.phi_AB_vectorfield)
        results.append(
            (
                case_id,
                flips_percentage_in_img,
                log_jac_std_in_img
            )
        )

        # Transform to displacement format that l2r evaluation script accepts
        disp = (phi_AB - identity)[0].cpu()

        network_shape_list = list(identity.shape[2:])

        dimension = len(network_shape_list)

        # We convert the displacement field into an itk Vector Image.
        scale = torch.Tensor(network_shape_list)

        for _ in network_shape_list:
            scale = scale[:, None]
        disp *= scale

        disp_itk_format = (
            disp.double()
            .numpy()[list(reversed(range(dimension)))]
            .transpose([3,2,1,0])
        )


    # Save to output folders
    disp_itk_format = nib.Nifti1Image(disp_itk_format, affine=np.eye(4))
    nib.save(disp_itk_format, f"{footsteps.output_dir}/submission/NLST/disp_{case_id}.nii.gz")

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