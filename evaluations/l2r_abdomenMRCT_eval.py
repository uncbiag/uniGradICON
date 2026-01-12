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
import utils
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

origin_shape = [1, 1, 192, 160, 192] # The submission system asks for 96 x 80 x 96
input_shape = [1, 1, 175, 175, 175]
clamp = [-1000, 1000]

args = parser.parse_args()
weights_path = args.weights_path
device = torch.device(f'cuda:{args.device}')
torch.cuda.set_device(device)

if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/L2R_abdomenMRCT")

logger = utils.Logger(f"{footsteps.output_dir}/output.txt")

os.makedirs(f"{footsteps.output_dir}/submission/submission/task_01", exist_ok=True)


def preprocess(img):
    im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
    img = torch.clip(img, im_min, im_max)
    img = (img-im_min) / (im_max-im_min)
    return img

net = make_network(input_shape, include_last_step=True)

logger.log(net.regis_net.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False))
net.to(device)
net.eval()

# L2R changed the idx of the image but kept the idx in the evaluation script the same as the old id.
cases_dict = {"0006":"0012", "0007":"0014", "0008":"0016"}

with open(f"{args.data_folder}/AbdomenMRCT_dataset.json", 'r') as data_info:
    data_info = json.loads(data_info.read())
test_cases = [[c["fixed"], c["moving"]] for c in data_info["registration_val"]]

spacing = 1.0 / (np.array(origin_shape[2::]) - 1)
identity = torch.from_numpy(icon.mermaidlite.identity_map_multiN(origin_shape, spacing)).to(device)

results = []
original_state_dict = copy.deepcopy(net.state_dict())
for (fixed_path, moving_path) in tqdm(test_cases):
    net.load_state_dict(original_state_dict)

    case_id = cases_dict[fixed_path.split('_')[1]]
    
    fixed = np.asarray(itk.imread(os.path.join(args.data_folder, fixed_path)))
    moving = np.asarray(itk.imread(os.path.join(args.data_folder, moving_path)))

    # Fixed: MRI
    fixed = torch.Tensor(np.array(fixed)).unsqueeze(0).unsqueeze(0)
    fixed = preprocess(fixed)
    fixed_in_net = F.interpolate(fixed, input_shape[2:], mode='trilinear', align_corners=False)
    
    # Move: CT
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
            disp.float()
            .numpy()[list(reversed(range(dimension)))]
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