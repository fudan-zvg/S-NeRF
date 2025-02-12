'''
Preprocess for foreground data.
This file generates the mask and masked img of vehicles.

Please copy this file to the root dir of vehicle data (e.g. "nerf_data_moving2_7") and run it.
'''

import numpy as np
import pickle
import os
from PIL import Image
from os.path import join

from ip_utils import scale_mat

#### Process the mask_images files. ####
im_list = sorted(os.listdir(join('./', 'mask_images', '0')))
for fname in im_list:
    path = join('./', 'mask_images', '0', fname)
    im_mat = np.array(Image.open(path))
    h, w, _ = im_mat.shape
    
    mask_mat = np.ones((h, w), dtype=bool)
    mask_mat[np.where(im_mat[..., 3] == 0)] = False  # [H, W]
    mask_mat = mask_mat.astype(np.uint8) * 255
    mask_mat = np.expand_dims(mask_mat, axis=2).repeat(3, axis=2)   # [H, W] -> [H, W, 3]
    
    im_mat = im_mat[..., :-1]   # [H, W, 4] -> [H, W, 3]
    im_mat[np.where(mask_mat == 0)] = 0

    # Modify the appearance.
    im_mat = np.power(im_mat, .93).astype(np.uint8)
    im_mat[:,:,0] = np.power(im_mat[:,:,0], .987).astype(np.uint8)
    im_mat[:,:,1] = np.power(im_mat[:,:,1], .993).astype(np.uint8)
        
    Image.fromarray(mask_mat).save(join('vehicles_mask', fname))
    Image.fromarray(im_mat).save(join('vehicles_img', fname))
#########################################