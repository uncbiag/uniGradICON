import itk
import numpy as np
import torch
import torch.nn.functional as F

import icon_registration
from icon_registration.itk_wrapper import create_itk_transform

import sys
sys.path.append("/playpen-raid2/lin.tian/projects/uniGradICON")
from evaluations.utils import compute_jacob_det_from_itk_transform, compute_jacob_det, Logger, compute_metrics, get_label_list

def itk_mean_dice(im1, im2):
    array1 = itk.array_from_image(im1)
    array2 = itk.array_from_image(im2)
    dices = []
    for index in range(1, max(np.max(array1), np.max(array2)) + 1):
        m1 = array1 == index
        m2 = array2 == index
        
        intersection = np.logical_and(m1, m2)
        
        d = 2 * np.sum(intersection) / (np.sum(m1) + np.sum(m2))
        dices.append(d)
    return np.mean(dices)

def register_pair(
    model, image_A, image_B, model_input_shape, device="cuda:0"
):

    assert isinstance(image_A, itk.Image)
    assert isinstance(image_B, itk.Image)

    model.bidir = True
    # send model to cpu or gpu depending on config- auto detects capability
    model.to(device)
    model.eval()

    A_npy = np.array(image_A)
    B_npy = np.array(image_B)

    assert(np.max(A_npy) != np.min(A_npy))
    assert(np.max(B_npy) != np.min(B_npy))
    # turn images into torch Tensors: add feature and batch dimensions (each of length 1)
    A_trch = torch.Tensor(A_npy).to(device)[None, None]
    B_trch = torch.Tensor(B_npy).to(device)[None, None]

    spacing = 1.0 / (np.array(model_input_shape[2::]) - 1)
    identity = torch.from_numpy(icon_registration.mermaidlite.identity_map_multiN(model_input_shape, spacing)).to(device)

    # Here we resize the input images to the shape expected by the neural network. This affects the
    # pixel stride as well as the magnitude of the displacement vectors of the resulting
    # displacement field, which create_itk_transform will have to compensate for.
    A_resized = F.interpolate(
        A_trch, size=model_input_shape[2:], mode="trilinear", align_corners=False
    )
    B_resized = F.interpolate(
        B_trch, size=model_input_shape[2:], mode="trilinear", align_corners=False
    )

    with torch.no_grad():
        _, pos_flow, neg_flow = model(A_resized, B_resized, registration=True)

    for i in range(len(model_input_shape[2:])):
        pos_flow[:, i, ...] = pos_flow[:, i, ...] / (model_input_shape[i+2] - 1)
        neg_flow[:, i, ...] = neg_flow[:, i, ...] / (model_input_shape[i+2] - 1)

    phi_AB = identity + pos_flow
    phi_BA = identity + neg_flow

    flip = icon_registration.losses.flips(phi_BA, in_percentage=True)

    # the parameters ident, image_A, and image_B are used for their metadata
    itk_transforms = (
        create_itk_transform(phi_AB, identity, image_A, image_B),
        create_itk_transform(phi_BA, identity, image_B, image_A),
        flip.detach().item(),
        phi_AB,
        phi_BA
    )
    

    return itk_transforms