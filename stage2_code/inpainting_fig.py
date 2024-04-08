import os
import cv2
import argparse
import numpy as np
import tqdm
from PIL import Image
from os.path import join

import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from inpaint import for_simnerf

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--wkdir', type=str, default="wkdir_0", required=True)  
    args = parser.parse_args()
    
    # First, copy the fuse data from out_dir_stage1.
    wkdir = args.wkdir
    if os.path.exists(join(wkdir, 'out_dir_stage2', 'image')):
        os.system("rm -r ./%s/out_dir_stage2/image" % wkdir)
        print("Remove ./%s/out_dir_stage2/image, copying..." % wkdir)
    if os.path.exists(join(wkdir, 'out_dir_stage2', 'image_after_inpaint')):
        os.system("rm -r ./%s/out_dir_stage2/image_after_inpaint" % wkdir)
        print("Remove ./%s/out_dir_stage2/image_after_inpaint, copying..." % wkdir)        
    os.system("cp -r ./%s/out_dir_stage1/fuse ./%s/out_dir_stage2/image" % (wkdir, wkdir))
    os.makedirs("./%s/out_dir_stage2/image_after_inpaint" % wkdir)
    print("Finish copying.")  
    
    # Process. First we need to get im/mask list.
    mask_dir = join(wkdir, 'out_dir_stage1', 'bound')
    mask_fname_list = sorted(os.listdir(mask_dir))
    mask_mat_list = []
    
    for fname in mask_fname_list:
        path = join(mask_dir, fname)
        mask_mat_list.append(np.array(Image.open(path))[..., 0])
    mask_mat = np.stack(mask_mat_list, axis=0)
    
    im_dir = join(wkdir, 'out_dir_stage2', 'image')
    im_fname_list = sorted(os.listdir(im_dir))
    im_mat_list = []
    
    for fname in im_fname_list:
        path = join(im_dir, fname)
        im_mat_list.append(np.array(Image.open(path)))
    im_mat = np.stack(im_mat_list, axis=0)
    
    # Inpaint.
    out_dir = join(wkdir, "out_dir_stage2", "image_after_inpaint")
    result_list = for_simnerf.choose_model_inpaint(im_mat, mask_mat, lama_or_AOT=0)
    ct = 0
    inpainted_im_mat_list = []
    for im_tensor in tqdm.tqdm(result_list):
        im_mat = im_tensor.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        im_mat = (im_mat * 255).astype(np.uint8)
        inpainted_im_mat_list.append(im_mat)
        # import pdb; pdb.set_trace()
        Image.fromarray(im_mat).save(join(out_dir, "%05d.png" % ct))
        ct += 1
        
    # Lightning.
    mask_dir = join(wkdir, 'out_dir_stage1', 'occluded_mask')
    mask_fname_list = sorted(os.listdir(mask_dir))
    mask_mat_list = []
    
    for fname in mask_fname_list:
        path = join(mask_dir, fname)
        one_mask_mat = np.array(Image.open(path))[..., 0]
        
        # Dilate.
        _, j_list = np.where(one_mask_mat > 0)
        if j_list.size != 0:
            mask_length = np.max(j_list) - np.min(j_list)
            r = int((mask_length / 80) ** .82)        
            kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1,r), max(1,r)))
            one_mask_mat = cv2.dilate(one_mask_mat, kernal, iterations=1)
            one_mask_mat = (one_mask_mat > 0).astype(np.uint8) * 255
        
        mask_mat_list.append(one_mask_mat)
    mask_mat = np.stack(mask_mat_list, axis=0)    
    
    inpainted_im_mat = np.stack(inpainted_im_mat_list, axis=0)
    
    out_dir = join(wkdir, "out_dir_stage2", "image")
    result_list = for_simnerf.choose_model_inpaint(inpainted_im_mat, mask_mat, lama_or_AOT=0, light=True)
    ct = 0
    for im_tensor in tqdm.tqdm(result_list):
        im_mat = im_tensor.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        im_mat = (im_mat * 255).astype(np.uint8)
        # inpainted_im_mat_list.append(im_mat)
        # import pdb; pdb.set_trace()
        Image.fromarray(im_mat).save(join(out_dir, "%05d.png" % ct))
        ct += 1    