import sys
sys.path.append('../')
from CartToPolar import *
from AugSurfSeg import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import os, random
import matplotlib.pyplot as plt
import cv2




ROW_LEN = 96
COL_LEN = 256
SIGMA = 10.


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



class IVUSDataset(Dataset):
    """convert 3d dataset to Dataset."""

    def __init__(self, img_np, gt_np, gaus_gt=True, transform=None, batch_nb=None):
        """
        Args:
            case_list (of dictionary): all  cases
            gaus_gt: output containes gaus_gt or not (default True)
            transform (callable, optional): Optional transform to be applied on a sample.
            batch_nb (int, optional): can use to debug.
        """
        self.img_np = np.load(img_np, mmap_mode='r')
        self.gt_np = np.load(gt_np, mmap_mode='r')
        self.bn = batch_nb
        self.gaus_gt = gaus_gt
        self.transform = transform

    def __len__(self):
        if self.bn is None:
            return self.img_np.shape[0]
        else:
            return self.bn

    def __getitem__(self, idx):
        img = self.img_np[idx,]
        gt = self.gt_np[idx,]
        # convert region gt to surface gt
        ret,thresh = cv2.threshold(imgray,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        gt_surf = contours[0]

        # convert to polar
        img_shape = img.shape[:-1]
        phy_radius = 0.5*np.sqrt(np.average(np.array(img_shape)**2)) - 1
        cartpolar = CartPolar(np.array(img_shape)/2.,
                              phy_radius, COL_LEN, ROW_LEN)
        polar_img = cartpolar.img2polar(img)
        polar_gt = cartpolar.gt2polar(gt_surf)
        input_img_gt = {'img': polar_img, 'gt': polar_gt}

        # apply augmentation transform
        if self.transform is not None:
            input_img_gt = self.transform(input_img_gt)

        # print(input_img_gt['gt'].shape)
        if self.gaus_gt:
            polar_gt_gaus = np.empty_like(polar_img[:,:,0])
            for i in range(COL_LEN):
                polar_gt_gaus[:, i] = LU_TABLE[int(
                    np.clip(np.around(input_img_gt['gt'][i]), 0, ROW_LEN-1)), ]
            input_img_gt['gaus_gt'] = polar_gt_gaus
        input_img_gt['img'] = np.expand_dims(input_img_gt['img'], axis=0)
        input_img_gt = {key:value.astype(np.float32) for (key, value) in input_img_gt.items()}
    
        return input_img_gt