# import math
import collections
from collections.abc import Sequence # for Python 3.10+
collections.Sequence = collections.abc.Sequence # for Python 3.10+
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

# This code is adapted from TransMorph: https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration

class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False): # class -> func()
        # image: nhwtc
        # shape: no first dim
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            # how to know  if the last dim is channel??
            # nhwtc vs nhwt??
            shape = im.shape[1:dim+1]
            # print(dim,shape) # 3, (240,240,155)
            self.sample(*shape)

        if isinstance(img, collections.Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)] # img:k=0,label:k=1

        return self.tf(img)

    def __str__(self):
        return 'Identity()'


class Seg_norm(Base):
    def __init__(self, ):
        a = None
        self.seg_table = np.array([0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255])
    def tf(self, img, k=0):
        if k == 0:
            return img
        img_out = np.zeros_like(img)
        for i in range(len(self.seg_table)):
            img_out[img == self.seg_table[i]] = i
        return img_out



class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)


class Compose(Base):
    def __init__(self, ops):
        if not isinstance(ops, collections.Sequence):
            ops = ops,
        self.ops = ops

    def sample(self, *shape):
        for op in self.ops:
            shape = op.sample(*shape)

    def tf(self, img, k=0):
        #is_tensor = isinstance(img, torch.Tensor)
        #if is_tensor:
        #    img = img.numpy()

        for op in self.ops:
            # print(op,img.shape,k)
            img = op.tf(img, k) # do not use op(img) here

        #if is_tensor:
        #    img = np.ascontiguousarray(img)
        #    img = torch.from_numpy(img)

        return img

    def __str__(self):
        ops = ', '.join([str(op) for op in self.ops])
        return 'Compose([{}])'.format(ops)

class IXIBrainInferDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg, path

    def __len__(self):
        return len(self.paths)

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def dice_val(y_pred, y_true, num_clus, target_labels):
    y_pred = torch.nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = torch.nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)
    dsc = torch.index_select(dsc, 1, target_labels)
    return torch.mean(torch.mean(dsc, dim=1))

def get_label_id(label_info_file):
    substruct = ['Left-Cerebral-White-Matter','Left-Cerebral-Cortex','Left-Lateral-Ventricle','Left-Inf-Lat-Vent','Left-Cerebellum-White-Matter','Left-Cerebellum-Cortex','Left-Thalamus-Proper*',
             'Left-Caudate','Left-Putamen','Left-Pallidum','3rd-Ventricle','4th-Ventricle','Brain-Stem','Left-Hippocampus','Left-Amygdala','CSF','Left-Accumbens-area','Left-VentralDC',
             'Left-vessel','Left-choroid-plexus','Right-Cerebral-White-Matter','Right-Cerebral-Cortex','Right-Lateral-Ventricle','Right-Inf-Lat-Vent','Right-Cerebellum-White-Matter',
             'Right-Cerebellum-Cortex','Right-Thalamus-Proper*','Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','Right-Amygdala','Right-Accumbens-area','Right-VentralDC',
             'Right-vessel','Right-choroid-plexus','5th-Ventricle','WM-hypointensities','non-WM-hypointensities','Optic-Chiasm','CC_Posterior','CC_Mid_Posterior','CC_Central','CC_Mid_Anterior,CC_Anterior']
    
    outstruct = ['Brain-Stem', 'Thalamus', 'Cerebellum-Cortex', 'Cerebral-White-Matter', 'Cerebellum-White-Matter', 'Putamen', 'VentralDC', 'Pallidum', 'Caudate', 'Lateral-Ventricle', 'Hippocampus',
             '3rd-Ventricle', '4th-Ventricle', 'Amygdala', 'Cerebral-Cortex', 'CSF', 'choroid-plexus']
    
    seg_table = np.array([0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255])
    
    # filter out the substructures
    filter_struct = []
    for out in outstruct:
        for sub in substruct:
            if out in sub:
                filter_struct.append(sub)
    
    # Build the mapping from label id to index in seg_table
    seg_dict = {}
    for i in range(len(seg_table)):
        seg_dict[seg_table[i]] = i

    import re
    with open(label_info_file, 'r') as f:
        lines = f.readlines()
        label_id = {}
        for line in lines[1:]:
            line = re.sub(' +', ' ',line).split(' ')
            try:
                int(line[0])
            except:
                continue
            if line[1] in filter_struct:
                label_id[seg_dict[int(line[0])]] = line[1]
    return label_id

