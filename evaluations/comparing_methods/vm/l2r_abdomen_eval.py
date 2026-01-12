import argparse
import json
import os

import footsteps
import itk
import numpy as np
import torch
import icon_registration as icon
import torch.nn.functional as F
from scipy.ndimage.interpolation import zoom as zoom
from tqdm import tqdm
import csv
import pandas as pd
import json
import utils


# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm 

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default="/playpen-raid2/lin.tian/data/learn2reg/L2R_Task3_AbdominalCT", help="the path to the folder containing learn2reg AbdomenCTCT dataset")
parser.add_argument("--weights_path", type=str, help="the path to the weights of the network")
parser.add_argument("--device", type=int, default=0, help="GPU ID.")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")

origin_shape = [1, 1, 256, 160, 192]
input_shape = [1, 1, 176, 176, 176]
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

net = vxm.networks.VxmDense.load(weights_path, device)
net.bidir = True
net.eval()
net.to(device)

os.makedirs(f"{footsteps.output_dir}/submission/task_03", exist_ok=True)

i

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

with open(f"{args.data_folder}/pairs_val.csv", 'r') as data_info:
    csv_reader = csv.reader(data_info)
    next(csv_reader)
    test_cases = [[f"Training/img/img{int(row[0]):04d}.nii.gz", f"Training/img/img{int(row[1]):04d}.nii.gz"] for row in csv_reader]

spacing = 1.0 / (np.array(input_shape[2::]) - 1)
identity = torch.from_numpy(icon.mermaidlite.identity_map_multiN(input_shape, spacing)).to(device)

spacing = 1.0 / (np.array(origin_shape[2::]) - 1)
origin_identity = torch.from_numpy(icon.mermaidlite.identity_map_multiN(origin_shape, spacing)).to(device)

results = []
for (fixed_path, moving_path) in tqdm(test_cases):

    case_id = f"{fixed_path.split('/')[-1].split('.')[0][3:]}_{moving_path.split('/')[-1].split('.')[0][3:]}"

    fixed = np.asarray(itk.imread(os.path.join(args.data_folder, fixed_path)))
    moving = np.asarray(itk.imread(os.path.join(args.data_folder, moving_path)))
    fixed_mask = np.asarray(itk.imread(os.path.join(args.data_folder, fixed_path.replace("img", "label"))))
    moving_mask = np.asarray(itk.imread(os.path.join(args.data_folder, moving_path.replace("img", "label"))))

    fixed = torch.Tensor(np.array(fixed)).unsqueeze(0).unsqueeze(0)
    fixed = (torch.clamp(fixed, clamp[0], clamp[1]) - clamp[0])/(clamp[1]-clamp[0])
    fixed_in_net = F.interpolate(fixed, input_shape[2:], mode='trilinear', align_corners=False)
    
    moving = torch.Tensor(np.array(moving)).unsqueeze(0).unsqueeze(0)
    moving = (torch.clamp(moving, clamp[0], clamp[1]) - clamp[0])/(clamp[1]-clamp[0])
    moving_in_net = F.interpolate(moving, input_shape[2:], mode='trilinear', align_corners=False)

    with torch.no_grad():
        _, pos_flow, _ = net(moving_in_net.to(device), fixed_in_net.to(device), registration=True)

        for i in range(3):
            pos_flow[:, i, ...] = pos_flow[:, i, ...] / (input_shape[i+2] - 1)

        phi_AB = identity + pos_flow

        phi_AB = torch.nn.functional.interpolate(phi_AB, size=origin_shape[2:], mode='trilinear', align_corners=False)
        
        # Compute the metrics
        flips_percentage_in_img, log_jac_std_in_img = utils.compute_jacob_det(phi_AB)
        results.append(
            (
                case_id,
                flips_percentage_in_img,
                log_jac_std_in_img
            )
        )

        # warped_seg = torch.nn.functional.grid_sample(torch.Tensor(moving_mask)[None,None].float().to(device), phi_AB.permute(0,2,3,4,1).flip(-1)*2.-1., mode='nearest', align_corners=False)

        # dices.append(mean_dice(fixed_mask.astype(int), warped_seg[0,0].cpu().numpy().astype(int)).item())

    # Transform to displacement format that l2r evaluation script accepts
    disp = (phi_AB- origin_identity)[0].cpu()

    network_shape_list = list(origin_identity.shape[2:])

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

# Prepare submission
import subprocess
subprocess.run(["cp", "-r", f"/playpen-raid2/lin.tian/projects/uniGradICON/evaluations/l2r_submission/task_01", f"{footsteps.output_dir}/submission/"])
subprocess.run(["cp", "-r", f"/playpen-raid2/lin.tian/projects/uniGradICON/evaluations/l2r_submission/task_02", f"{footsteps.output_dir}/submission/"])
subprocess.run(["cp", "-r", f"/playpen-raid2/lin.tian/projects/uniGradICON/evaluations/l2r_submission/task_04", f"{footsteps.output_dir}/submission/"])
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