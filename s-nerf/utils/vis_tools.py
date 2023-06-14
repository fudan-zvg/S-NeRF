import torch
import numpy as np
import os.path as osp
import cv2
from pyquaternion import Quaternion
import imageio
from PIL import Image

MAX_WIDTH= 1600
MAX_HEIGHT = 900

def visualize_depth(im_mat, bds):
    '''
    Plot the velodyne points in the im image.
    args:
        im_mat: An numpy array.
    returns:
        im_mat: An Image instance.
    '''
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).reshape(256, 3))
    grey_to_color_map = dict(zip(gray_values, color_values))
    
    near, far = bds
    im_mat = (im_mat-near) / (far-near) * 255

    h, w = im_mat.shape
    canvas = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            if(im_mat[i, j] < 0 or im_mat[i, j] > 255):
                continue
            canvas[i, j] = np.array(grey_to_color_map[round(im_mat[i, j])])

    return canvas[:,:,::-1]

def visualize_gray(im_mat):
    '''
    Plot the velodyne points in the im image.
    args:
        im_mat: An numpy array.
    returns:
        im_mat: An Image instance.
    '''
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_HOT).reshape(256, 3))
    grey_to_color_map = dict(zip(gray_values, color_values))

    im_mat = im_mat / im_mat.max() * 255

    h, w = im_mat.shape
    canvas = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            canvas[i, j] = np.array(grey_to_color_map[round(im_mat[i, j])])

    return canvas[:,:,::-1]