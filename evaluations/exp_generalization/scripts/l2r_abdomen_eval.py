import argparse
import csv
import os

import footsteps
import icon_registration as icon
import itk
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.append("/playpen-raid2/lin.tian/projects/uniGradICON/evaluations")
import utils
from scipy.ndimage.interpolation import zoom as zoom
from tqdm import tqdm

from unigradicon import finetune_execute, make_network
from model_inshape import inshape_dict

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str, help="the path to the weights of the network")
parser.add_argument("--data_folder", type=str, help="the path to the folder containing learn2reg AbdomenCTCT dataset")
parser.add_argument("--io_steps", type=int, default=0, help="Steps for IO")
parser.add_argument("--device", type=int, default=0, help="GPU ID.")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")
parser.add_argument("--model_type", type=str, default="lung", help="the type of the model. (lung, brain, knee)}")


origin_shape = [1, 1, 256, 160, 192]
input_shape = [1, 1, 175, 175, 175]
clamp = [-1000, 1000]

args = parser.parse_args()
weights_path = args.weights_path
device = torch.device(f'cuda:{args.device}')
torch.cuda.set_device(device)

if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/L2R_abdomen")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

os.makedirs(f"{footsteps.output_dir}/submission/task_03", exist_ok=True)

input_shape = inshape_dict[f"{args.model_type}_model"]
net = make_network(input_shape, include_last_step=True)

logger.log(net.regis_net.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True))
net.to(device)
net.eval()

import json
import os

import itk
import nibabel as nib


def mean_dice(im1, im2):
    array1 = im1
    array2 = im2
    dices = []
    for index in range(1, max(np.max(array1), np.max(array2)) + 1):
        m1 = array1 == index
        m2 = array2 == index
        
        intersection = np.logical_and(m1, m2)
        
        d = 2 * np.sum(intersection) / (np.sum(m1) + np.sum(m2))
        dices.append(d)
    return np.mean(dices)

# with open(f"{args.data_folder}/AbdomenCTCT_dataset.json", 'r') as data_info:
#     data_info = json.loads(data_info.read())
# test_cases = [[c["fixed"], c["moving"]] for c in data_info["registration_val"]]

with open(f"{args.data_folder}/pairs_val.csv", 'r') as data_info:
    csv_reader = csv.reader(data_info)
    next(csv_reader)
    test_cases = [[f"Training/img/img{int(row[0]):04d}.nii.gz", f"Training/img/img{int(row[1]):04d}.nii.gz"] for row in csv_reader]

spacing = 1.0 / (np.array(origin_shape[2::]) - 1)
identity = torch.from_numpy(icon.mermaidlite.identity_map_multiN(origin_shape, spacing)).to(device)

flips = []
original_state_dict = net.state_dict()
for (fixed_path, moving_path) in tqdm(test_cases):
    # Restore net weight in case we ran IO
    # net.load_state_dict(original_state_dict)
    
    fixed = np.asarray(itk.imread(os.path.join(args.data_folder, fixed_path)))
    moving = np.asarray(itk.imread(os.path.join(args.data_folder, moving_path)))

    fixed = torch.Tensor(np.array(fixed)).unsqueeze(0).unsqueeze(0)
    fixed = (torch.clamp(fixed, clamp[0], clamp[1]) - clamp[0])/(clamp[1]-clamp[0])
    fixed_in_net = F.interpolate(fixed, input_shape[2:], mode='trilinear', align_corners=False)
    
    moving = torch.Tensor(np.array(moving)).unsqueeze(0).unsqueeze(0)
    moving = (torch.clamp(moving, clamp[0], clamp[1]) - clamp[0])/(clamp[1]-clamp[0])
    moving_in_net = F.interpolate(moving, input_shape[2:], mode='trilinear', align_corners=False)

    if args.io_steps > 0:
        finetune_execute(net, moving_in_net.to(device), fixed_in_net.to(device), args.io_steps)

    with torch.no_grad():
        net(moving_in_net.to(device), fixed_in_net.to(device))

        # phi_AB and phi_BA are [1, 3, H, W, D] pytorch tensors representing the forward and backward
        # maps computed by the model
        phi_AB = net.phi_AB(identity)

        flips.append(icon.losses.flips(phi_AB, in_percentage=True).item())

        # Transform to displacement format that l2r evaluation script accepts
        disp = (phi_AB- identity)[0].cpu()

        network_shape_list = list(identity.shape[2:])

        dimension = len(network_shape_list)

        # We convert the displacement field into an itk Vector Image.
        scale = torch.Tensor(network_shape_list)

        for _ in network_shape_list:
            scale = scale[:, None]
        disp *= scale

        disp_itk_format = (
            disp.float()
            .numpy()[list(reversed(range(dimension)))]
            .transpose([0,3,2,1])
        )

        # Save to output folders in the format required by l2r evaluation script
        disp_x = zoom(disp_itk_format[0], 0.5, order=2).astype('float16')
        disp_y = zoom(disp_itk_format[1], 0.5, order=2).astype('float16')
        disp_z = zoom(disp_itk_format[2], 0.5, order=2).astype('float16')
        disp = np.array((disp_x, disp_y, disp_z))
        np.savez_compressed(f"{footsteps.output_dir}/submission/task_03/disp_{fixed_path.split('/')[-1].split('.')[0][3:]}_{moving_path.split('/')[-1].split('.')[0][3:]}.npz", disp)
        # np.savez_compressed(f"{footsteps.output_dir}/submission/task_03/disp_{fixed_path.split('_')[1]}_{moving_path.split('_')[1]}.npz", disp)

        # Save to output folders
        # disp_itk_format = nib.Nifti1Image(disp_itk_format, affine=np.eye(4))
        # nib.save(disp_itk_format, f"{footsteps.output_dir}/disp_{fixed_path.split('_')[1]}_{moving_path.split('_')[1]}.nii.gz")

# Prepare submission
import subprocess

subprocess.run(["cp", "-r", f"/playpen-raid2/lin.tian/projects/uniGradICON/evaluations/l2r_submission/task_01", f"{footsteps.output_dir}/submission/"])
subprocess.run(["cp", "-r", f"/playpen-raid2/lin.tian/projects/uniGradICON/evaluations/l2r_submission/task_02", f"{footsteps.output_dir}/submission/"])
subprocess.run(["cp", "-r", f"/playpen-raid2/lin.tian/projects/uniGradICON/evaluations/l2r_submission/task_04", f"{footsteps.output_dir}/submission/"])
subprocess.call("zip -r submission.zip ./*", shell=True, cwd=f"{footsteps.output_dir}/submission/")   

logger.log(f"Mean folds percentage: {np.mean(flips)}")