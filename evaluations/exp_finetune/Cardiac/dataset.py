import torch
import os
import torch.nn.functional as F
import random
import numpy as np
import itk
from tqdm import tqdm
import glob

class ACDCMMDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path="",
            data_num=1000,
            desired_shape=None,
            device="cpu"
    ):
        case_list = glob.glob(data_path + "/*_img.pt")
        
        random.seed(42)
        random.shuffle(case_list)
        
        if data_num < len(case_list):
            case_list = case_list[:data_num]

        print("Loading cardiac data...")
        self.imgs = []
        for img_path in tqdm(case_list):
            self.imgs.append(
                self.process(torch.load(img_path, map_location="cpu").float()[None], desired_shape, device)[0]
                ) # 2xHxWxD, ES and ED
        
        self.data_num = len(self.imgs)
        print(f"Using {len(self.imgs)} images for training.")
    
    def process(self, img, desired_shape=None, device="cpu"):
        # We only process the shape here. Because the intensity of the 
        # processed cardiac images is already in the range of [0, 1]
        if desired_shape is not None and desired_shape != list(img.shape[2:]):
            img = img.to(device)
            img = F.interpolate(img, desired_shape, mode="trilinear") 
        return img.cpu()
    
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        '''
        Return
        img_a: 1xHxWxD
        img_b: 1xHxWxD
        '''
        img = self.imgs[idx]
        if random.random() < 0.5:
            img_a = img[0:1]
            img_b = img[1:2]
        else:
            img_a = img[1:2]
            img_b = img[0:1]
        return img_a, img_b