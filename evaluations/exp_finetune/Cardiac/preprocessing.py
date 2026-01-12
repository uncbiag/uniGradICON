import numpy as np
import os
import itk
import tqdm
import glob
import nibabel as nib
import torch
import torch.nn.functional as F
import pandas as pd

target_size = [175, 175, 175]  # Target size for resampling


def process_shape_and_intensity(img, desired_shape=None, device="cpu"):
    img = img.to(device)
    im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
    img = torch.clip(img, im_min, im_max)
    img = (img-im_min) / (im_max-im_min)
    if desired_shape is not None:
        img = F.interpolate(img, desired_shape, mode="trilinear") 
    return img.cpu()

def process_MM_files(img_path_list, df, output_dir):

    for img_path in tqdm.tqdm(img_path_list):
        case_id = img_path.split("/")[-1].split("_")[0]

        ES_slice_id = df[df["External code"] == case_id]["ES"].values[0]
        ED_slice_id = df[df["External code"] == case_id]["ED"].values[0]

        img = torch.from_numpy(nib.load(img_path).get_fdata())
        ES_img = process_shape_and_intensity(img[:,:,:,ES_slice_id][None, None].contiguous(), desired_shape=target_size, device="cuda")
        ED_img = process_shape_and_intensity(img[:,:,:,ED_slice_id][None, None].contiguous(), desired_shape=target_size, device="cuda")
        img = torch.concat([ES_img, ED_img], dim=1)[0]

        torch.save(img, f"{output_dir}/imgs/MM_{case_id}_img.pt")

def process_ACDC_files(case_list, output_dir):
    for case in case_list:
        case_id = case.split("/")[-2]
        
        with open(os.path.join(case, "Info.cfg")) as f:
            for line in f.readlines():
                if "ES" in line:
                    ES_id = int(line.split(":")[-1][:-1])
                    ES_img = process_shape_and_intensity(
                        torch.from_numpy(
                            nib.load(os.path.join(case, f"{case_id}_frame{ES_id:02d}.nii.gz")).get_fdata()
                        )[None, None].contiguous(), desired_shape=target_size, device="cuda")
                elif "ED" in line:
                    ED_id = int(line.split(":")[-1][:-1])
                    ED_img = process_shape_and_intensity(
                        torch.from_numpy(
                            nib.load(os.path.join(case, f"{case_id}_frame{ED_id:02d}.nii.gz")).get_fdata()
                        )[None, None].contiguous(), desired_shape=target_size, device="cuda")
        
        img = torch.concat([ES_img, ED_img], dim=1)[0]
        torch.save(img, f"{output_dir}/imgs/ACDC_{case_id}_img.pt")

if __name__ == "__main__":
    # TODO: Set output path
    output_dir = "./processed_data/cardiac_mri"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(f"{output_dir}/imgs"):
        os.makedirs(f"{output_dir}/imgs")

    # TODO: Set MM cardiac data path
    data_path = "./original_data/MM_cardiac/Training/Labeled"
    data_list = glob.glob(data_path + "/*/*_sa.nii.gz")
    df = pd.read_csv("./original_data/MM_cardiac/211230_MMs_Dataset_information_diagnosis_opendataset.csv")
    print("Processing MM labeled files...")
    process_MM_files(data_list, df, output_dir)

    # TODO: Set MM cardiac data path
    data_path = "./original_data/MM_cardiac/Training/Unlabeled"
    data_list = glob.glob(data_path + "/*/*_sa.nii.gz")
    print("Processing MM unlabeled files...")
    process_MM_files(data_list, df, output_dir)

    # TODO: Set MM cardiac data path
    print("Processing ACDC files...")
    ACDC_cases = glob.glob('./original_data/ACDC/database/training/patient*/')
    process_ACDC_files(ACDC_cases, output_dir)