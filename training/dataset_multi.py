import torch
import os
import torch.nn.functional as F
import random
import numpy as np
import itk
import glob
import SimpleITK as sitk
from tqdm import tqdm
import blosc
import json

blosc.set_nthreads(1)
DATASET_DIR = "./data/multiGradICON/"

class COPDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scale="2xdown",
        data_path=f"{DATASET_DIR}/half_res_preprocessed_transposed_SI",
        ROI_only=False,
        data_num=-1,
        desired_shape=None,
        device="cpu",
        return_labels=False
    ):
        self.imgs = torch.load(f"{data_path}/lungs_train_{scale}_scaled", map_location="cpu")
        self.data_num = data_num
        self.desired_shape = desired_shape
        self.device = device
        self.return_labels = return_labels
        self.modalities = ['ct']
        self.anatomies = ['lung']
        self.region_num = 1

        print("Processing COPD data.")
        if ROI_only:
            segs = torch.load(
                f"{data_path}/lungs_seg_train_{scale}_scaled", map_location="cpu"
            )[: self.data_num]
            self.imgs = list(map(lambda x: (self.pack_and_process_image(x[0][0], x[1][0]), self.pack_and_process_image(x[0][1], x[1][1])), tqdm(zip(self.imgs, segs))))
        else:
            self.imgs = list(map(lambda x: (self.pack_and_process_image(x[0]), self.pack_and_process_image(x[1])), tqdm(self.imgs)))

    def pack_and_process_image(self, img, seg=None):
        processed_image = self.process(img, self.desired_shape, self.device, seg)[0]
        array_image = processed_image.numpy()
        return blosc.pack_array(array_image)
        
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
        img_a, img_b = random.choice(self.imgs)
        
        if self.return_labels:
            return blosc.unpack_array(img_a), blosc.unpack_array(img_b), blosc.unpack_array(img_a), blosc.unpack_array(img_b)
        else:
            return blosc.unpack_array(img_a), blosc.unpack_array(img_b)

class BratsRegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path=f"{DATASET_DIR}/BraTS-Reg/BraTSReg_Training_Data_v3/",
        data_num=1000,
        desired_shape=None,
        device="cpu",
        return_labels=False,
        randomization = 'random'
    ):
        
        folders = sorted(glob.glob(data_path + '*/'))
        self.pre_images, self.post_images = {'t1': [], 't1ce': [], 't2': [], 'flair': []}, {'t1': [], 't1ce': [], 't2': [], 'flair': []}
        self.desired_shape = desired_shape
        self.device = device
        self.return_labels = return_labels
        self.randomization = randomization
        assert self.randomization in ['random', 'fixed']
        self.anatomies = ['brain']
        self.region_num = 1

        for folder in tqdm(folders):
            t1 = itk.imread(glob.glob(folder + 'BraTSReg_*_00_*_t1.nii.gz')[0])
            t1ce = itk.imread(glob.glob(folder + 'BraTSReg_*_00_*_t1ce.nii.gz')[0])
            t2 = itk.imread(glob.glob(folder + 'BraTSReg_*_00_*_t2.nii.gz')[0])
            flair = itk.imread(glob.glob(folder + 'BraTSReg_*_00_*_flair.nii.gz')[0])
            
            self.pre_images['t1'].append(self.pack_and_process_image(t1))
            self.pre_images['t1ce'].append(self.pack_and_process_image(t1ce))
            self.pre_images['t2'].append(self.pack_and_process_image(t2))
            self.pre_images['flair'].append(self.pack_and_process_image(flair))
            
            t1 = itk.imread(glob.glob(folder + 'BraTSReg_*_01_*_t1.nii.gz')[0])
            t1ce = itk.imread(glob.glob(folder + 'BraTSReg_*_01_*_t1ce.nii.gz')[0])
            t2 = itk.imread(glob.glob(folder + 'BraTSReg_*_01_*_t2.nii.gz')[0])
            flair = itk.imread(glob.glob(folder + 'BraTSReg_*_01_*_flair.nii.gz')[0])
            
            self.post_images['t1'].append(self.pack_and_process_image(t1))
            self.post_images['t1ce'].append(self.pack_and_process_image(t1ce))
            self.post_images['t2'].append(self.pack_and_process_image(t2))
            self.post_images['flair'].append(self.pack_and_process_image(flair))
            
        self.data_num = data_num
        self.image_num = len(self.pre_images['t1'])
        self.modalities = list(self.pre_images.keys())
        
    def pack_and_process_image(self, image):
        processed_image = self.process(torch.tensor(np.asarray(image))[None, None], self.desired_shape, self.device)[0]
        array_image = processed_image.numpy()
        return blosc.pack_array(array_image)

    def process(self, img, desired_shape=None, device="cpu"):
        img = img.to(device).float()
        im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
        img = torch.clip(img, im_min, im_max)
        img = (img-im_min) / (im_max-im_min)
        if desired_shape is not None:
            img = F.interpolate(img, desired_shape, mode="trilinear") 
        return img.cpu().float()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        idx1 = random.randint(0, self.image_num-1)
        
        img_a = self.pre_images[random.choice(self.modalities)][idx1]
        img_b = self.post_images[random.choice(self.modalities)][idx1]
        
        if not self.return_labels:
            return blosc.unpack_array(img_a), blosc.unpack_array(img_b)
        
        if self.randomization == 'random':
            label_a = self.pre_images[random.choice(self.modalities)][idx1]
            label_b = self.post_images[random.choice(self.modalities)][idx1]
        else:
            modality = random.choice(self.modalities)
            label_a = self.pre_images[modality][idx1]
            label_b = self.post_images[modality][idx1]

        return blosc.unpack_array(img_a), blosc.unpack_array(img_b), blosc.unpack_array(label_a), blosc.unpack_array(label_b)

class L2rAbdomenDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path=f"{DATASET_DIR}/AbdomenCTCT",
            data_num=1000,
            desired_shape=None,
            device="cpu",
            return_labels=False,
            randomization = 'random',
            augmentation = True
    ):
        cases = list(map(lambda x: os.path.join(f"{data_path}/imagesTr", x), os.listdir(f"{data_path}/imagesTr/")))
        self.desired_shape = desired_shape
        self.device = device
        self.return_labels = return_labels
        self.randomization = randomization
        assert self.randomization in ['random', 'fixed']
        self.anatomies = ['abdomen']
        self.region_num = 1
        
        self.imgs = {'ct' : []}
        if augmentation:
            self.imgs['1-ct'] = []
        
        print("Processing L2R Abdomen data.")
        for i in tqdm(range(len(cases))):
            case_path = cases[i]
            self.imgs['ct'].append(self.pack_and_process_image(case_path))
            if augmentation:
                self.imgs['1-ct'].append(self.pack_and_process_image(case_path, invert=True))
        
        self.img_num = len(self.imgs)
        self.data_num = data_num
        self.modalities = list(self.imgs.keys())
        
    def pack_and_process_image(self, case_path, invert=False):
        processed_image = self.process(torch.tensor(np.asarray(itk.imread(case_path)))[None, None], self.desired_shape, self.device)[0]
        array_image = processed_image.numpy()
        if invert:
            array_image = 1 - array_image
        return blosc.pack_array(array_image)
        
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
        img_a = self.imgs[random.choice(self.modalities)][idx_a]
        img_b = self.imgs[random.choice(self.modalities)][idx_b]
        
        if not self.return_labels:
            return blosc.unpack_array(img_a), blosc.unpack_array(img_b)
        
        if self.randomization == 'random':
            label_a = self.imgs[random.choice(self.modalities)][idx_a]
            label_b = self.imgs[random.choice(self.modalities)][idx_b]
        else:
            modality = random.choice(self.modalities)
            label_a = self.imgs[modality][idx_a]
            label_b = self.imgs[modality][idx_b]
            
        return blosc.unpack_array(img_a), blosc.unpack_array(img_b), blosc.unpack_array(label_a), blosc.unpack_array(label_b)

class HCPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scale="2xdown",
        data_path=f"{DATASET_DIR}/ICON_brain_preprocessed_data",
        data_num=1000,
        desired_shape=None,
        device="cpu",
        return_labels=False,
        randomization = 'random'
    ):
        self.desired_shape = desired_shape
        self.device = device
        self.return_labels = return_labels
        self.randomization = randomization
        assert self.randomization in ['random', 'fixed']
        self.anatomies = ['brain']
        self.region_num = 1
        
        imgsT1 = torch.load(
            f"{data_path}/brain_train_{scale}_scaled_T1", map_location="cpu"
        )
        imgsT2 = torch.load(
            f"{data_path}/brain_train_{scale}_scaled_T2", map_location="cpu"
        )
        imgsT1 = list(map(lambda x: self.pack_and_process_image(x), tqdm(imgsT1)))
        imgsT2 = list(map(lambda x: self.pack_and_process_image(x), tqdm(imgsT2)))
        
        self.imgs = {'T1': imgsT1, 'T2': imgsT2}
    
        self.img_num = len(imgsT1)
        self.data_num = data_num
        self.modalities = list(self.imgs.keys())
        
    def pack_and_process_image(self, image):
        processed_image = self.process(image, self.desired_shape, self.device)[0]
        array_image = processed_image.numpy()
        return blosc.pack_array(array_image)
            
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

        img_a = self.imgs[random.choice(self.modalities)][idx_a]
        img_b = self.imgs[random.choice(self.modalities)][idx_b]
        
        if not self.return_labels:
            return blosc.unpack_array(img_a), blosc.unpack_array(img_b)
        
        if self.randomization == 'random':
            label_a = self.imgs[random.choice(self.modalities)][idx_a]
            label_b = self.imgs[random.choice(self.modalities)][idx_b]
        else:
            modality = random.choice(self.modalities)
            label_a = self.imgs[modality][idx_a]
            label_b = self.imgs[modality][idx_b]
            
        return blosc.unpack_array(img_a), blosc.unpack_array(img_b), blosc.unpack_array(label_a), blosc.unpack_array(label_b)

class ABCDFAMDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        phase="train",
        data_path=f"{DATASET_DIR}/dti_scalars",
        data_num=1000,
        desired_shape=None,
        device="cpu",
        return_labels=False,
        randomization = 'random'
    ):
        self.desired_shape = desired_shape
        self.device = device
        self.return_labels = return_labels
        self.randomization = randomization
        assert self.randomization in ['random', 'fixed']
        self.images = {}
        self.anatomies = ['brain']
        self.region_num = 1
            
        md_files = sorted(glob.glob(f'{data_path}/md/' + '*.nii.gz'))
        fa_files = sorted(glob.glob(f'{data_path}/fa/' + '*.nii.gz'))
        
        if phase == "train":
            fa_files = fa_files[10:]
            md_files = md_files[10:]
        elif phase == "val":
            fa_files = fa_files[:10]
            md_files = md_files[:10]
            
        fa_ids = {x.split('/')[-1].split('.')[0].split('_')[0].split('-')[1] : x for x in fa_files}
        md_ids = {x.split('/')[-1].split('.')[0].split('_')[0].split('-')[1] : x for x in md_files}
        
        for fa_id in tqdm(fa_ids):
            if fa_id not in md_ids:
                continue
            
            fa = itk.imread(fa_ids[fa_id])
            md = itk.imread(md_ids[fa_id])
            self.images[fa_id] = {'FA': self.pack_and_process_image(fa), 'MD': self.pack_and_process_image(md)}

        self.data_num = data_num
        self.image_ids = list(self.images.keys())
        self.modalities = ['FA', 'MD']
        
    def pack_and_process_image(self, image):
        processed_image = self.process(self.process(torch.tensor(np.asarray(image))[None, None], self.desired_shape, self.device)[0])
        array_image = processed_image.numpy()
        return blosc.pack_array(array_image)

    def process(self, img, desired_shape=None, device="cpu"):
        img = img.to(device).float()
        im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
        img = torch.clip(img, im_min, im_max)
        img = (img-im_min) / (im_max-im_min)
        if desired_shape is not None:
            img = F.interpolate(img, desired_shape, mode="trilinear") 
        return img.cpu().float()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        index1 = random.choice(self.image_ids)
        image1 = self.images[index1][random.choice(self.modalities)]
        
        index2 = random.choice(self.image_ids)
        image2 = self.images[index2][random.choice(self.modalities)]
        
        if not self.return_labels:
            return blosc.unpack_array(image1), blosc.unpack_array(image2)
        
        if self.randomization == 'random':
            label1 = self.images[index1][random.choice(self.modalities)]
            label2 = self.images[index2][random.choice(self.modalities)]
        else:
            modality = random.choice(self.modalities)
            label1 = self.images[index1][modality]
            label2 = self.images[index2][modality]

        return blosc.unpack_array(image1), blosc.unpack_array(image2), blosc.unpack_array(label1), blosc.unpack_array(label2)
       
class ABCDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        phase="train",
        data_path=f"{DATASET_DIR}",
        data_num=1000,
        desired_shape=None,
        device="cpu",
        return_labels=False,
    ):

        md_path = f'{data_path}/dti_scalars/md/'
        md_files = sorted(glob.glob(md_path + '*.nii.gz'))
        
        fa_path = f'{data_path}/dti_scalars/fa/'
        fa_files = sorted(glob.glob(fa_path + '*.nii.gz'))
        
        mri_path = f'{data_path}/structural_mri/'
        mri_files = sorted(glob.glob(mri_path + '*_oriented_stripped.nii.gz'))
        
        if phase == "train":
            fa_files = fa_files[10:]
            md_files = md_files[10:]
            mri_files = mri_files[10:]
        elif phase == "val":
            fa_files = fa_files[:10]
            md_files = md_files[:10]
            mri_files = mri_files[:10]
            
        fa_ids = {x.split('/')[-1].split('.')[0].split('_')[0].split('-')[1] : x for x in fa_files}
        md_ids = {x.split('/')[-1].split('.')[0].split('_')[0].split('-')[1] : x for x in md_files}
        mri_ids = {x.split('/')[-1].split('.')[0].split('_')[0].split('-')[1] : x for x in mri_files}
        
        self.desired_shape = desired_shape
        self.device = device
        self.return_labels = return_labels
        self.images = {'FA': [], 'MD': [], 'T1': [], 'T2': []}
        self.anatomies = ['brain']
        self.region_num = 1
        
        for id in tqdm(fa_ids):
            fa = itk.imread(fa_ids[id])
            self.images['FA'].append(self.pack_and_process_image(fa))
            md = itk.imread(md_ids[id])
            self.images['MD'].append(self.pack_and_process_image(md))

        for mri_id in tqdm(mri_ids):
            mri = mri_ids[mri_id].replace('T1w', 'modality').replace('T2w', 'modality')
            if not os.path.exists(mri.replace('modality', 'T1w')) or not os.path.exists(mri.replace('modality', 'T2w')):
                continue
            
            t1_mri = itk.imread(mri.replace('modality', 'T1w'))
            self.images['T1'].append(self.pack_and_process_image(t1_mri))
            
            t2_mri = itk.imread(mri.replace('modality', 'T2w'))
            self.images['T2'].append(self.pack_and_process_image(t2_mri))
            
        self.data_num = data_num
        self.modalities = list(self.images.keys())
        
    def pack_and_process_image(self, image):
        processed_image = self.process(self.process(torch.tensor(np.asarray(image))[None, None], self.desired_shape, self.device)[0])
        array_image = processed_image.numpy()
        return blosc.pack_array(array_image)

    def process(self, img, desired_shape=None, device="cpu"):
        img = img.to(device).float()
        im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
        img = torch.clip(img, im_min, im_max)
        img = (img-im_min) / (im_max-im_min)
        if desired_shape is not None:
            img = F.interpolate(img, desired_shape, mode="trilinear") 
        return img.cpu().float()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        image1 = np.random.choice(self.images[np.random.choice(self.modalities)])
        image2 = np.random.choice(self.images[np.random.choice(self.modalities)])
        
        if not self.return_labels:
            return blosc.unpack_array(image1), blosc.unpack_array(image2)
        else:
            return blosc.unpack_array(image1), blosc.unpack_array(image2), blosc.unpack_array(image1), blosc.unpack_array(image2)
    
    
class OAIMMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path=f"{DATASET_DIR}/oai",
        data_num=1000,
        desired_shape=None,
        device="cpu",
        return_labels=False,
    ):
        self.desired_shape = desired_shape
        self.device = device
        self.return_labels = return_labels
        self.anatomies = ['knee']
        self.region_num = 1
                
        dataset_dess = torch.load(f"{data_path}/dess_images.pt", map_location="cpu")
        dataset_T2 = torch.load(f"{data_path}t2_images.pt", map_location="cpu")
        
        self.images = {'DESS': [], 'T2': []}
        
        self.images['DESS'] = list(map(lambda x: self.pack_and_process_image(x), tqdm(dataset_dess)))
        self.images['T2'] = list(map(lambda x: self.pack_and_process_image(x), tqdm(dataset_T2)))
        
        self.data_num = data_num
        self.modalities = list(self.images.keys())
        
    def pack_and_process_image(self, image):
        processed_image = self.process(image, self.desired_shape, self.device)[0]
        array_image = processed_image.numpy()
        return blosc.pack_array(array_image)
        
    def process(self, img, desired_shape=None, device="cpu"):
        img = img.to(device).float()
        im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
        img = torch.clip(img, im_min, im_max)
        img = (img-im_min) / (im_max-im_min)
        if desired_shape is not None:
            img = F.interpolate(img, desired_shape, mode="trilinear") 
        return img.cpu()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        modality_1, modality_2 = random.choices(self.modalities, k=2)
        img_a = random.choice(self.images[modality_1])
        img_b = random.choice(self.images[modality_2])
        
        if self.return_labels:        
            return blosc.unpack_array(img_a), blosc.unpack_array(img_b), blosc.unpack_array(img_a), blosc.unpack_array(img_b)
        else:
            return blosc.unpack_array(img_a), blosc.unpack_array(img_b)

class L2rMRCTDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path=f"{DATASET_DIR}/AbdomenMRCT/",
            data_num=1000,
            desired_shape=None,
            device="cpu",
            phase = "train",
            augmentation = True,
            return_labels=False
    ):
        #inter-patient
        self.phase = phase
        self.device = device
        self.return_labels = return_labels
        self.anatomies = ['abdomen']
        self.region_num = 1
        
        with open(f"{data_path}/AbdomenMRCT_dataset.json", 'r') as data_info:
            data_info = json.loads(data_info.read())
            
        if self.phase == "train":
            mr_samples = [c["image"] for c in data_info["training"]["0"]] #mr
            ct_samples = [c["image"] for c in data_info["training"]["1"]] #ct
        else:
            mr_samples = [c["fixed"] for c in data_info["registration_test"]]
            ct_samples = [c["moving"] for c in data_info["registration_test"]]
            
        self.images = {'mr' : [], 'ct' : []}
        
        if augmentation:
            self.images['1-ct'] = []
        
        for path in tqdm(mr_samples):
            image = np.asarray(itk.imread(os.path.join(data_path, path)))
            image = torch.Tensor(np.array(image)).unsqueeze(0).unsqueeze(0)
            image = self.process_mr(image, desired_shape, device)[0]
            self.images['mr'].append(self.pack(image))
            
        for path in tqdm(ct_samples):
            image = np.asarray(itk.imread(os.path.join(data_path, path)))
            image = torch.Tensor(np.array(image)).unsqueeze(0).unsqueeze(0)
            image = self.process_ct(image, desired_shape, device)[0]
            self.images['ct'].append(self.pack(image))
            if augmentation:
                self.images['1-ct'].append(self.pack(1-image))
        
        self.data_num = data_num
        self.modalities = list(self.images.keys())
        
    def pack(self, image):
        return blosc.pack_array(image.numpy())
    
    def process_label(self, label, desired_shape=None, device="cpu"):
        label = label.to(device)
        if desired_shape is not None:
            label = F.interpolate(label, desired_shape, mode="nearest")
        return label.cpu()
    
    def process_ct(self, img, desired_shape=None, device="cpu"):
        img = img.to(device)
        img = (torch.clip(img.float(), -1000, 1000)+1000)/2000
        if desired_shape is not None:
            img = F.interpolate(img, desired_shape, mode="trilinear") 
        return img.cpu()
    
    def process_mr(self, img, desired_shape=None, device="cpu"):
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
        modality1 = random.choice(self.modalities)
        modality2 = random.choice(self.modalities)
        
        idx1 = random.randint(0, len(self.images[modality1])-1)
        idx2 = random.randint(0, len(self.images[modality2])-1)
        
        img_a = self.images[modality1][idx1]
        img_b = self.images[modality2][idx2]
        
        if self.return_labels:
            return blosc.unpack_array(img_a), blosc.unpack_array(img_b), blosc.unpack_array(img_a), blosc.unpack_array(img_b)
        else:
            return blosc.unpack_array(img_a), blosc.unpack_array(img_b)


class UKBiobankDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path=f"{DATASET_DIR}/uk-biobank/",
            data_num=1000,
            desired_shape=None,
            device="cpu",
            phase = "train",
            return_labels=False,
            randomization = 'random'
    ):
        #contains 6 regions of the body - each region contains a list of images
        fat_weighted_regions = torch.load(data_path + 'regions_fat.pt', map_location='cpu')
        water_weighted_regions = torch.load(data_path + 'regions_water.pt', map_location='cpu')
            
        self.desired_shape = desired_shape
        self.device = device
        self.return_labels = return_labels
        self.randomization = randomization
        assert self.randomization in ['random', 'fixed']
        self.anatomies = ['abdomen', 'lung', 'knee']
        
        self.images = {'fat': fat_weighted_regions, 'water': water_weighted_regions}
        
        for region in tqdm(range(6)):
            if phase == "train":
                self.images['fat'][region] = list(map(lambda x: self.pack_and_process_image(x), self.images['fat'][region][10:]))
                self.images['water'][region] = list(map(lambda x: self.pack_and_process_image(x), self.images['water'][region][10:]))
            else:
                self.images['fat'][region] = list(map(lambda x: self.pack_and_process_image(x), self.images['fat'][region][:10]))
                self.images['water'][region] = list(map(lambda x: self.pack_and_process_image(x), self.images['water'][region][:10]))
            
        self.data_num = data_num
        self.modalities = ['fat', 'water']
        self.region_num = 6
        
    def pack_and_process_image(self, image):
        processed_image = self.process(image, self.desired_shape, self.device)[0]
        array_image = processed_image.numpy()
        return blosc.pack_array(array_image)
    
    def process(self, img, desired_shape=None, device="cpu"):
        img = torch.tensor(sitk.GetArrayFromImage(img)[None, None].astype(np.float32))
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
        region = random.randint(0, self.region_num-1)
        modality1 = random.choice(self.modalities)
        modality2 = random.choice(self.modalities)
        
        index1 = random.randint(0, len(self.images[modality1][region])-1)
        index2 = random.randint(0, len(self.images[modality2][region])-1)
        
        img_a = self.images[modality1][region][index1]
        img_b = self.images[modality2][region][index2]
        
        if not self.return_labels:
            return blosc.unpack_array(img_a), blosc.unpack_array(img_b)
    
        if self.randomization == 'random':
            label1 = self.images[random.choice(self.modalities)][region][index1]
            label2 = self.images[random.choice(self.modalities)][region][index2]
        else:
            modality = random.choice(self.modalities)
            label1 = self.images[modality][region][index1]
            label2 = self.images[modality][region][index2]
        
        return blosc.unpack_array(img_a), blosc.unpack_array(img_b), blosc.unpack_array(label1), blosc.unpack_array(label2)
   
 
class PancreasDataset(torch.utils.data.Dataset):
    def __init__(self,         
            phase="train",
            data_path=f"{DATASET_DIR}/pancreas/",
            data_num=1000,
            desired_shape=(175, 175, 175),
            device="cpu",
            return_labels=False,
        ):

        with open(f"{data_path}/{phase}_set.txt", 'r') as f:
            self.img_list = [line.strip() for line in f]

        self.desired_shape = desired_shape
        self.data_num = data_num
        self.img_dict = {}
        self.return_labels = return_labels
        self.modalities = ['ct', 'cbct']
        self.region_num = 1
        self.anatomies = ['abdomen']

        for idx in range(len(self.img_list)):
            ith_info = self.img_list[idx].split(" ")
            ct_img_name = ith_info[0]
            cb_img_name = ith_info[1]

            ct_img_itk = sitk.ReadImage(ct_img_name)
            cb_img_itk = sitk.ReadImage(cb_img_name)

            ct_img_arr = sitk.GetArrayFromImage(ct_img_itk)
            cb_img_arr = sitk.GetArrayFromImage(cb_img_itk)

            ct_img_arr, cb_img_arr = self.process_training_data(ct_img_arr, cb_img_arr)

            self.img_dict[ct_img_name] = blosc.pack_array(ct_img_arr)
            self.img_dict[cb_img_name] = blosc.pack_array(cb_img_arr)

    def __len__(self):
        return self.data_num

    def process(self, img, desired_shape=None, device="cpu"):
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).cpu()
        img = img.to(device).float()
        img = (torch.clip(img.float(), -1000, 1000)+1000)/2000
        if desired_shape is not None:
            img = F.interpolate(img, desired_shape, mode="trilinear") 
        return img.squeeze().numpy().cpu().float()     

    def process_training_data(self, ct_img_arr, cb_img_arr):
        ct_img_arr = self.process(ct_img_arr, self.desired_shape)
        cb_img_arr = self.process(cb_img_arr, self.desired_shape)

        return ct_img_arr, cb_img_arr
    
    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.img_list))
        ith_info = self.img_list[idx].split(" ")
        ct_img_name = ith_info[0]
        cb_img_name = ith_info[1]

        ct_img_arr = blosc.unpack_array(self.img_dict[ct_img_name])
        cb_img_arr = blosc.unpack_array(self.img_dict[cb_img_name])

        if not self.return_labels:
            return ct_img_arr, cb_img_arr
        else:
            return ct_img_arr, cb_img_arr, ct_img_arr, cb_img_arr

class L2rThoraxCBCTDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path=f"{DATASET_DIR}/ThoraxCBCT",
            data_num=1000,
            desired_shape=None,
            device="cpu",
            return_labels=False,
    ):

        with open(f"{data_path}/ThoraxCBCT_dataset.json", 'r') as data_info:
            data_info = json.loads(data_info.read())
        cases = [f"{data_path}/{c['image']}" for c in data_info["training"]]
        
        self.modalities = {'0000': [] , '0001': [], '0002': []}
        self.desired_shape = desired_shape
        self.device = device
        self.return_labels = return_labels
        self.anatomies = ['lung']
        self.region_num = 1
        
        for case in cases:
            modality = case.split('/')[-1].split('_')[-1].split('.')[0]
            self.modalities[modality].append(self.pack_and_process_image(itk.imread(case)))
        
        self.data_num = data_num
        
    def pack_and_process_image(self, image):
        processed_image = self.process(self.process(torch.tensor(np.asarray(image))[None, None], self.desired_shape, self.device)[0])
        array_image = processed_image.numpy()
        return blosc.pack_array(array_image)

    def process(self, img, desired_shape=None, device="cpu"):
        img = img.to(device).float()
        img = (torch.clip(img.float(), -1000, 1000)+1000)/2000
        if desired_shape is not None:
            img = F.interpolate(img, desired_shape, mode="trilinear") 
        return img.cpu().float()      

    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        modality1 = random.choice(list(self.modalities.keys()))
        modality2 = random.choice(list(self.modalities.keys()))
        
        patient_id = random.randint(0, len(self.modalities[modality1])-1)
        
        image1 = self.modalities[modality1][patient_id]
        image2 = self.modalities[modality2][patient_id]
        
        if not self.return_labels:
            return blosc.unpack_array(image1), blosc.unpack_array(image2)
        else:
            return blosc.unpack_array(image1), blosc.unpack_array(image2), blosc.unpack_array(image1), blosc.unpack_array(image2)