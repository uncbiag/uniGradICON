import torch
import os
import torch.nn.functional as F
import random
import numpy as np
import itk
import glob
import SimpleITK as sitk
from tqdm import tqdm

DATASET_DIR = "./data/uniGradICON/"

class COPDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        phase="train",
        scale="2xdown",
        data_path=f"{DATASET_DIR}/half_res_preprocessed_transposed_SI",
        ROI_only=False,
        data_num=-1,
        desired_shape=None,
        device="cpu"
    ):
        if phase == "debug":
            phase = "train"
        self.imgs = torch.load(
            f"{data_path}/lungs_{phase}_{scale}_scaled", map_location="cpu"
        )
        if data_num <= 0 or data_num > len(self.imgs):
            self.data_num = len(self.imgs)
        else:
            self.data_num = data_num
        self.imgs = self.imgs[: self.data_num]

        # Normalize to [0,1]
        print("Processing COPD data.")
        if ROI_only:
            segs = torch.load(
                f"{data_path}/lungs_seg_{phase}_{scale}_scaled", map_location="cpu"
            )[: self.data_num]
            self.imgs = list(map(lambda x: (self.process(x[0][0], desired_shape, device, x[1][0])[0],self.process(x[0][1], desired_shape, device, x[1][1])[0]), tqdm(zip(self.imgs, segs))))
        else:
            self.imgs = list(map(lambda x: (self.process(x[0], desired_shape, device)[0],self.process(x[1], desired_shape, device)[0]), tqdm(self.imgs)))

    
    def process(self, img, desired_shape=None, device="cpu", seg=None):
        img = img.to(device)
        im_min, im_max = torch.min(img), torch.max(img)
        img = (img-im_min) / (im_max-im_min)
        if seg is not None:
            seg = seg.to(device)
            img = (img * seg).float()
        if desired_shape is not None:
            img = F.interpolate(img, desired_shape, mode="trilinear") 
        return img.cpu()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        img_a, img_b = self.imgs[idx]
        return img_a, img_b


class OAIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        phase="train",
        scale="2xdownsample",
        data_path=f"{DATASET_DIR}/OAI",
        data_num=1000,
        desired_shape=None,
        device="cpu"
    ):
        if phase == "debug":
            phase = "train"
        if phase == "test":
            phase = "train"
            print(
                "WARNING: There is no validation set for OAI. Using train data for test set."
            )
        self.imgs = torch.load(
            f"{data_path}/knees_big_{scale}_train_set", map_location="cpu"
        )

        print("Processing OAI data.")
        self.imgs = list(map(lambda x: self.process(x, desired_shape, device)[0], tqdm(self.imgs)))

        self.img_num = len(self.imgs)

        self.data_num = data_num
    
    def process(self, img, desired_shape=None, device="cpu"):
        img = img.to(device)
        im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
        img = torch.clip(img, im_min, im_max)
        img = (img-im_min) / (im_max-im_min)
        
        if desired_shape is not None:
            img = F.interpolate(img, desired_shape, mode="trilinear") 
        return img.cpu()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        idx_a = random.randint(0, self.img_num - 1)
        idx_b = random.randint(0, self.img_num - 1)
        img_a = self.imgs[idx_a]
        img_b = self.imgs[idx_b]
        return img_a, img_b


class HCPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        phase="train",
        scale="2xdown",
        data_path=f"{DATASET_DIR}/HCP",
        data_num=1000,
        desired_shape=None,
        device="cpu"
    ):
        if phase == "debug":
            phase = "train"
        if phase == "test":
            phase = "train"
            print(
                "WARNING: There is no validation set for OAI. Using train data for test set."
            )
        self.imgs = torch.load(
            f"{data_path}/brain_train_{scale}_scaled", map_location="cpu"
        )
        print("Processing HCP data.")
        self.imgs = list(map(lambda x: self.process(x, desired_shape, device)[0], tqdm(self.imgs)))

        self.img_num = len(self.imgs)

        self.data_num = data_num
    
    def process(self, img, desired_shape=None, device="cpu"):
        img = img.to(device)
        im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
        img = torch.clip(img, im_min, im_max)
        img = (img-im_min) / (im_max-im_min)
        if desired_shape is not None:
            img = F.interpolate(img, desired_shape, mode="trilinear") 
        return img.cpu()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        idx_a = random.randint(0, self.img_num - 1)
        idx_b = random.randint(0, self.img_num - 1)
        img_a = self.imgs[idx_a]
        img_b = self.imgs[idx_b]
        return img_a, img_b


class L2rAbdomenDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path=f"{DATASET_DIR}/AbdomenCTCT",
            data_num=1000,
            desired_shape=None,
            device="cpu"
    ):
        cases = list(map(lambda x: os.path.join(f"{data_path}/imagesTr", x), os.listdir(f"{data_path}/imagesTr/")))
        self.imgs = []
        print("Processing L2R Abdomen data.")
        for i in tqdm(range(len(cases))):
            case_path = cases[i]
            self.imgs.append(self.process(torch.tensor(np.asarray(itk.imread(case_path)))[None, None], desired_shape, device)[0])
        
        self.img_num = len(self.imgs)

        self.data_num = data_num
        
    def process(self, img, desired_shape=None, device="cpu"):
        img = img.to(device)
        img = (torch.clip(img.float(), -1000, 1000)+1000)/2000
        if desired_shape is not None:
            img = F.interpolate(img, desired_shape, mode="trilinear") 
        return img.cpu()
    
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        idx_a = random.randint(0, self.img_num - 1)
        idx_b = random.randint(0, self.img_num - 1)
        img_a = self.imgs[idx_a]
        img_b = self.imgs[idx_b]
        return img_a, img_b
    

class L2rThoraxCBCTDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path=f"{DATASET_DIR}/ThoraxCBCT",
            data_num=1000,
            desired_shape=None,
            device="cpu"
    ):
        import json
        with open(f"{data_path}/ThoraxCBCT_dataset.json", 'r') as data_info:
            data_info = json.loads(data_info.read())
        cases = [[f"{data_path}/{c['moving']}", f"{data_path}/{c['fixed']}"] for c in data_info["training_paired_images"]]
        self.imgs = []
        print("Processing L2R ThoraxCBCT data.")
        for i in tqdm(range(len(cases))):
            moving_path, fixed_path = cases[i]
            self.imgs.append(
                (
                    self.process(torch.tensor(np.asarray(itk.imread(moving_path)))[None, None], desired_shape, device)[0],
                    self.process(torch.tensor(np.asarray(itk.imread(fixed_path)))[None, None], desired_shape, device)[0]
                ))
        
        if data_num < len(self.imgs):
            self.imgs = self.imgs[:data_num]
        self.data_num = len(self.imgs)
        
    def process(self, img, desired_shape=None, device="cpu"):
        img = img.to(device)
        img = (torch.clip(img.float(), -1000, 1000)+1000)/2000
        if desired_shape is not None:
            img = F.interpolate(img, desired_shape, mode="trilinear") 
        return img.cpu()
    
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        img_a, img_b = self.imgs[idx]
        return img_a, img_b
    
        
class ACDCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path=f"{DATASET_DIR}/ACDC",
        desired_shape=None
    ):
        self.imgs = []
        training_files = sorted(glob(f"{data_path}/database/training/*/*4d.nii.gz"))

        for file in training_files:
            self.imgs.append(
                self.process(
                    torch.tensor(
                        sitk.GetArrayFromImage(sitk.ReadImage(file, sitk.sitkFloat32))
                    )[None],
                    desired_shape)
            )
        
        self.img_num = len(self.imgs)
        
    
    def process(self, img, desired_shape=None):
        img = img / torch.amax(img, dim=(1,2,3), keepdim=True)

        # Pad the image
        pad_size = (img.shape[2] - img.shape[1]) // 2
        img = torch.pad(img, (0, 0, 0, 0, pad_size, pad_size), "constant", 0)

        if desired_shape is not None:
            img = F.interpolate(img, desired_shape, mode="trilinear") 
        return img
    
    def __len__(self):
        return self.img_num
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        img_count = img.shape[0]
        idx_a = random.randint(0, img_count - 1)
        idx_b = random.randint(0, img_count - 1)
        img_a = img[idx_a]
        img_b = img[idx_b]
        return img_a, img_b


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    datasets = [COPDDataset, OAIDataset, HCPDataset, L2rAbdomenDataset]

    for dataset in datasets:
        data = dataset(desired_shape=(64, 64, 64))
        data = DataLoader(data, batch_size=3)
        print(next(iter(data))[0].shape)
