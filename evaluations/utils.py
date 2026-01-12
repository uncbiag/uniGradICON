import itk
import numpy as np
import torch
import sys
from surface_distance import compute_surface_distances, compute_robust_hausdorff

class Logger:
    def __init__(self, file):
        self.file = open(file, "w")
    
    def log(self, *val):
        print(*val)
        print(*val, file=self.file)
    
    def close(self):
        self.file.close()

def itk_mean_dice(im1, im2, labels, return_details=False):
    array1 = itk.array_from_image(im1).astype(np.uint8)
    array2 = itk.array_from_image(im2).astype(np.uint8)
    
    dices = {}
    for index in labels:
        m1 = array1 == index
        m2 = array2 == index
        
        intersection = np.logical_and(m1, m2)
        
        d = 2 * np.sum(intersection) / (np.sum(m1) + np.sum(m2))
        dices[index] = d
    
    if return_details:
        return np.mean(list(dices.values())), dices
    else:
        return np.mean(list(dices.values()))

def mean_dice(target, source, warped, labels):
    dices = []
    for index in labels:
        m1 = target == index
        m2 = source == index

        if target.sum() == 0 or source.sum() == 0:
            dices.append(np.nan)
        else:
            m2 = warped == index
            intersection = np.logical_and(m1, m2)
            
            d = 2 * np.sum(intersection) / (np.sum(m1) + np.sum(m2))
            dices.append(d)
    
    return np.nanmean(dices), dices

def compute_hd95(target, source, warped, labels, spacing):
    '''
    Compute surface distances between two images.
    Args:
        im1: First image (Numpy array).
        im2: Second image (Numpy array).
        spacing: Spacing of the images (tuple or list).

    '''
    hd95 = []
    for i in labels:
        seg1 = target == i
        seg2 = source == i

        if seg1.sum() == 0 or seg2.sum() == 0:
            hd95.append(np.nan)
        else:
            seg2 = warped == i
            hd95.append(compute_robust_hausdorff(
                compute_surface_distances(
                    seg1, seg2, spacing), 95))
    m_hd95 = np.nanmean(hd95)
    return m_hd95, hd95

def compute_metrics(target, source, warped, labels):
    '''
    Compute specified metrics between two images.
    Args:
        im1: First image (itk image).
        im2: Second image (itk image).
        spacing: Spacing of the images (tuple or list).
        labels: List of labels to compute metrics for.
        metrics: List of metrics to compute (default is ["dice"]).

    '''
    target_array = itk.array_from_image(target).astype(np.uint8)
    source_array = itk.array_from_image(source).astype(np.uint8)
    warped_array = itk.array_from_image(warped).astype(np.uint8)
    im1_spacing = np.array(target.GetSpacing())[::-1].copy()  # Reverse to match numpy order
    im2_spacing = np.array(warped.GetSpacing())[::-1].copy()  # Reverse to match numpy order
    assert (im1_spacing == im2_spacing).all(), "Images must have the same spacing. If not, resample one of the images to match the spacing of the other."

    dices, dice_details = mean_dice(target_array, source_array, warped_array, labels)
    hd95, hd95_details = compute_hd95(target_array, source_array, warped_array, labels, im1_spacing)
    metrics = {
        "dice": dices,
        "hd95": hd95,
        "dice_details": dice_details,
        "hd95_details": hd95_details,
        "labels": labels
    }

    return metrics

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


def compute_jacob_det_from_itk_transform(transform, reference, mask=None):
    jacob = itk.displacement_field_jacobian_determinant_filter(
        itk.transform_to_displacement_field_filter(
            transform,
            use_reference_image=True,
            reference_image=reference
        )
    )

    jacob_np = np.array(jacob)
    if mask is not None:
        mask = np.array(mask)
        mask[mask>0] = 1
        jacob_np = np.ma.MaskedArray(jacob_np, mask)
    flips_percentage = np.mean(jacob_np<0) * 100.

    # Following the implementation in Learn2Reg
    # https://github.com/MDL-UzL/L2R/blob/main/evaluation/evaluation.py#L139
    log_jac_det_std = np.log((jacob_np+3).clip(1e-9, 1e9)).std()

    return flips_percentage, log_jac_det_std

def get_label_list(case_id):
    dataset = case_id.split(".")[0]
    if dataset in segmentation_labels:
        return list(segmentation_labels[dataset].keys())
    else:
        print("Error. Did not find the label list.")


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
    },
    "HCP":{
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 12,
        13: 13,
        14: 14,
        15: 15,
        16: 16,
        17: 17,
        18: 18,
        19: 19,
        20: 20,
        21: 21,
        22: 22,
        23: 23,
        24: 24,
        25: 25,
        26: 26,
        27: 27,
        28: 28
    },
    "OAI":{
        1: 1,
        2: 2
    },
    "IXI":{
        1: 'Left-Cerebral-White-Matter', 2: 'Left-Cerebral-Cortex', 3: 'Left-Lateral-Ventricle', 5: 'Left-Cerebellum-White-Matter', 6: 'Left-Cerebellum-Cortex', 7: 'Left-Thalamus-Proper*', 8: 'Left-Caudate', 9: 'Left-Putamen', 10: 'Left-Pallidum', 11: '3rd-Ventricle', 12: '4th-Ventricle', 13: 'Brain-Stem', 14: 'Left-Hippocampus', 15: 'Left-Amygdala', 16: 'CSF', 18: 'Left-VentralDC', 20: 'Left-choroid-plexus', 21: 'Right-Cerebral-White-Matter', 22: 'Right-Cerebral-Cortex', 23: 'Right-Lateral-Ventricle', 25: 'Right-Cerebellum-White-Matter', 26: 'Right-Cerebellum-Cortex', 27: 'Right-Thalamus-Proper*', 28: 'Right-Caudate', 29: 'Right-Putamen', 30: 'Right-Pallidum', 31: 'Right-Hippocampus', 32: 'Right-Amygdala', 34: 'Right-VentralDC', 36: 'Right-choroid-plexus'
    }
}

def finetune_execute(model, image_A, image_B, steps, lr=2e-5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss_tuple = model(image_A, image_B)
        print(loss_tuple)
        loss_tuple[0].backward()
        optimizer.step()
    with torch.no_grad():
        loss = model(image_A, image_B)
    model.eval()
    return loss