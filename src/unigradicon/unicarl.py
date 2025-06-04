import itk
import os
from datetime import datetime

import footsteps
import numpy as np
import torch
import torch.nn.functional as F

import icon_registration as icon
import icon_registration.network_wrappers as network_wrappers
import icon_registration.networks as networks
from icon_registration import config
from icon_registration.losses import ICONLoss, to_floats
from icon_registration.mermaidlite import compute_warped_image_multiNC
import icon_registration.itk_wrapper

input_shape = [1, 1, 160, 160, 160]

import icon_registration.unicarl.register as register
import icon_registration.unicarl.affine_decomposition as affine_decomposition

from unigradicon import maybe_cast

def get_unicarl():
    from os.path import exists
    weights_location = "network_weights/unicarl_preview/longleaf_unicarl_8"
    if not exists(weights_location):
        print("Downloading pretrained unicarl model")
        import urllib.request
        import os
        download_path = "https://github.com/uncbiag/uniGradICON/releases/download/v1.0.4/longleaf_unicarl_8"
        os.makedirs("network_weights/unicarl_preview/", exist_ok=True)
        urllib.request.urlretrieve(download_path, weights_location)
    print(f"Loading weights from {weights_location}")
    net = register.get_model(path=weights_location)
    net.to(config.device)
    net.eval()
    return net


def main():
    import itk
    import argparse
    parser = argparse.ArgumentParser(description="Register two images using uniCARL.")
    parser.add_argument("--fixed", required=True, type=str,
                         help="The path of the fixed image.")
    parser.add_argument("--moving", required=True, type=str,
                         help="The path of the fixed image.")
    parser.add_argument("--transform_out", required=True,
                         type=str, help="The path to save the transform.")
    parser.add_argument("--warped_moving_out", required=False,
                        default=None, type=str, help="The path to save the warped image.")
    parser.add_argument("--io_iterations", required=False,
                         default="50", help="The number of IO iterations. Default is 50. Set to 'None' to disable IO.")
    args = parser.parse_args()
    

    net = get_unicarl() 

    fixed = itk.imread(args.fixed)
    moving = itk.imread(args.moving)
    

    if args.io_iterations == "None":
        io_iterations = None
    else:
        io_iterations = int(args.io_iterations)

    phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(
        net,
        register.preprocess(moving),
        register.preprocess(fixed),
        finetune_steps=io_iterations)
    phi_AB = affine_decomposition.decompose_icon_itk_transform(phi_AB)

    itk.transformwrite([phi_AB], args.transform_out)

    if args.warped_moving_out:
        moving, maybe_cast_back = maybe_cast(moving)
        interpolator = itk.LinearInterpolateImageFunction.New(moving)
        warped_moving_image = itk.resample_image_filter(
                moving,
                transform=phi_AB,
                interpolator=interpolator,
                use_reference_image=True,
                reference_image=fixed
                )
        warped_moving_image = maybe_cast_back(warped_moving_image)
        itk.imwrite(warped_moving_image, args.warped_moving_out)

