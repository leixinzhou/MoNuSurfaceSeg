import sys
sys.path.append('../')
from cartpolar import *
from AugSurfSeg import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import os, random
import matplotlib.pyplot as plt
import cv2




ROW_LEN = 32
COL_LEN = 64
SIGMA = 6.


def gaus_pdf(x_arry, mean, sigma, A=1.):
    pdf_array = np.empty_like(x_arry, dtype=np.float16)
    for i in range(x_arry.shape[0]):
        pdf_array[i] = A * np.exp((-(x_arry[i] - mean)**2)/(2*sigma**2))
    pdf_array = np.exp(pdf_array) / np.sum(np.exp(pdf_array), axis=0)
    return pdf_array


# define a look-up table
# lookup table
LU_TABLE = np.empty((ROW_LEN, ROW_LEN), dtype=np.float16)
x_range = np.arange(ROW_LEN).astype(np.float16)
for i in range(ROW_LEN):
    prob = gaus_pdf(x_range, i, SIGMA)
    LU_TABLE[i, ] = prob


class MoNuDataset(Dataset):
    """convert 3d dataset to Dataset."""

    def __init__(self, polar_img_np, polar_gt_np, img_np=None, gt_np=None, gaus_gt=True, transform=None, batch_nb=None):
        """
        Args:
            case_list (of dictionary): all  cases
            gaus_gt: output containes gaus_gt or not (default True)
            transform (callable, optional): Optional transform to be applied on a sample.
            batch_nb (int, optional): can use to debug.
        """
        self.img_np = np.load(polar_img_np, mmap_mode='r')
        self.gt_np = np.load(polar_gt_np, mmap_mode='r')
        self.bn = batch_nb
        self.gaus_gt = gaus_gt
        self.transform = transform
        self.cart_img_np = img_np
        self.cart_gt_np = gt_np
        if self.cart_img_np is None:
            pass
        else:
            self.cart_img_np = np.load(img_np, mmap_mode='r')
        if self.cart_gt_np is None:
            pass
        else:
            self.cart_gt_np = np.load(gt_np, mmap_mode='r')

    def __len__(self):
        if self.bn is None:
            return self.img_np.shape[0]
        else:
            return self.bn

    def __getitem__(self, idx):
        polar_img = self.img_np[idx,].copy()
        polar_gt = self.gt_np[idx,].copy()
        
        # normalize image 
        for i in range(3):
            mean, std = np.mean(polar_img[i,]), np.std(polar_img[i,])
            polar_img[i,] = (polar_img[i,]-mean)*1./std
        
        input_img_gt = {'img': polar_img, 'gt': polar_gt}

        # apply augmentation transform
        if self.transform is not None:
            input_img_gt = self.transform(input_img_gt)

        # print(input_img_gt['gt'].shape)
        if self.gaus_gt:
            polar_gt_gaus = np.zeros((ROW_LEN, COL_LEN), dtype=np.float32)
            for i in range(COL_LEN):
                pos = int(np.clip(np.around(input_img_gt['gt'][i]), 0, ROW_LEN-1))
                # print(pos)
                polar_gt_gaus[:, i] = LU_TABLE[pos, ]
            input_img_gt['gaus_gt'] = polar_gt_gaus
                # plt.imshow(np.transpose(self.img_np[idx,], (1,2,0)))
                # plt.plot(polar_gt)
                # plt.show()
        # input_img_gt['img'] = np.expand_dims(input_img_gt['img'], axis=0)
        # plt.imshow(polar_gt_gaus, cmap='gray')
        # plt.show()
        if self.cart_img_np is None:
            pass
        else:
            input_img_gt['cart_img'] = self.cart_img_np[idx,].copy()
        if self.cart_gt_np is None:
            pass
        else:
            input_img_gt['cart_gt'] = self.cart_gt_np[idx,].copy()
        input_img_gt = {key:torch.from_numpy(value.astype(np.float32)) for (key, value) in input_img_gt.items()}
    
        return input_img_gt