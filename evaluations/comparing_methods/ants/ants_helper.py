import itk
import numpy as np
import SimpleITK as sitk
import torch
import ants
import sys

def copy_reference_image_info(array, reference_image):
    """
    Create an ITK image from a numpy array with metadata from a reference image.
    
    Args:
        array: NumPy array with image data
        reference_image: ITK image to copy metadata from
    
    Returns:
        ITK image with data from array and metadata from reference_image
    """
    # Create ITK image from array
    output_image = itk.GetImageFromArray(array)
    
    # Copy metadata from reference image
    output_image.SetOrigin(reference_image.GetOrigin())
    output_image.SetSpacing(reference_image.GetSpacing())
    output_image.SetDirection(reference_image.GetDirection())
    
    return output_image

def compute_jacob_det_for_ants(ants_transform, fixed, mask=None):
    '''
    Args:
        transformation: the ants transform object
    '''
    jac = ants.create_jacobian_determinant_image(fixed, ants_transform, do_log=False, geom=False)
    jacob_np = jac.numpy()
    if mask is not None:
        mask = np.array(mask)[0,0, :-1, :-1, :-1]
        mask[mask>0] = 1
        jacob_np = np.ma.MaskedArray(jacob_np, mask)
    flips_percentage = np.mean(jacob_np<0) * 100.

    # Following the implementation in Learn2Reg
    # https://github.com/MDL-UzL/L2R/blob/main/evaluation/evaluation.py#L139
    log_jac_det_std = np.log((jacob_np+3).clip(1e-9, 1e9)).std() 

    return flips_percentage, log_jac_det_std

# This solution is provided by BailiangJ in https://github.com/ANTsX/ANTsPy/issues/427
def load_flow(flow_path:str):
    # displacement fields of ANTs and SimpleITK are both in Physical Point coordinate
    disp = sitk.ReadImage(flow_path)
    direction = torch.tensor(disp.GetDirection()).reshape(3,3)
    spacing = torch.diag(torch.tensor(disp.GetSpacing()))
    # the computed Affine matrix exclude the Origin
    # since we are transforming the displacement vector in Physical Point coordinate
    # to Image Index coordinate, the Origin is not needed
    affine = torch.matmul(direction, spacing)
    # mapping from Image Index coordinate to Physical Point coordinate, so we need the inverse
    affine_inv = torch.linalg.inv(affine)
    
    # sitk: (x,y,z) -> numpy:(z,y,x)
    disp_arr = sitk.GetArrayFromImage(disp)
    disp_arr = np.transpose(disp_arr, axes=(3,2,1,0)) #(3,H,W,D)
    disp_tensor = torch.from_numpy(disp_arr).float()
    
    # if pkg == "niftyreg":
    #         nifty_to_sitk = torch.tensor([-1.0,0,0,0,-1.0,0,0,0,1.0]).reshape(3,3)
    #         # from nifty space to sitk space
    #         # the x, y axes are mirrored
    #         disp_tensor = torch.einsum("ij,jhwd->ihwd", nifty_to_sitk, disp_tensor)
    
    # Physical Point space displacement to Image Index space displacement
    disp_tensor = torch.einsum("ij,jhwd->ihwd", affine_inv, disp_tensor)
    return disp_tensor.unsqueeze(0)

def run_ants(fixed, moving, type_of_transform, tmp_folder, fixed_mask=None, moving_mask=None, return_disp_tensor=True):
    reg_res = ants.registration(fixed, moving, type_of_transform=type_of_transform, outprefix=tmp_folder, verbose=False)

    if return_disp_tensor:
        # The composite transform will be saved in the tmp_folder
        composed = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg_res['fwdtransforms'], compose=tmp_folder)

        # # save the displacement field
        # disp = ants.image_read(composed)
        # disp_arr = disp.numpy()
        # disp = ants.from_numpy(disp_arr,origin=disp.origin,spacing=disp.spacing,direction=disp.direction,has_components=disp.has_components,is_rgb=disp.is_rgb)
        # ants.image_write(disp, f"{tmp_folder}/ants_flow.nii.gz")

        # disp_tensor = load_flow(f"{tmp_folder}/ants_flow.nii.gz")
        disp_tensor = load_flow(composed)
    else:
        disp_tensor = None

    if fixed_mask is not None:
        warped_seg = ants.apply_transforms(fixed=fixed_mask, moving=moving_mask, transformlist=reg_res['fwdtransforms'], interpolator="nearestNeighbor")
    else:
        warped_seg = None
            
    return disp_tensor, warped_seg, reg_res