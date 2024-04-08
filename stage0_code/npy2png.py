import os
import time
import yaml
import shutil
import imageio
import argparse
import numpy as np
from os.path import join
from PIL import Image, ImageDraw
import sys
sys.path.append('.')

def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print("Create dir: %s" % dir)
    else:
        f_list = os.listdir(dir)
        for f in f_list:
            if os.path.isdir(os.path.join(dir, f)):
                shutil.rmtree(os.path.join(dir, f))
            else:
                os.remove(join(dir, f))
        print("Remove all files in %s" % dir)

result_dir = join('./raw_data/background')

sub_result_dirs = sorted(os.listdir(result_dir))

for sub_dir in sub_result_dirs:
    sub_path = join(result_dir, sub_dir)
    rgb_f, depth_f, sem_f = join(sub_path, 'rgb.npy'), join(sub_path, 'depth.npy'), join(sub_path, 'semantic.npy')
    
    if os.path.exists(rgb_f):
        rgb_file = np.load(rgb_f)
        n_images = rgb_file.shape[0]
        out_dir = join(sub_path, 'rgb')
        create_dir(out_dir)
        for idx in range(n_images):
            Image.fromarray(rgb_file[idx, ...]).save(join(out_dir, '%05d.png' % idx))
    
    if os.path.exists(depth_f):
        depth_file = np.load(depth_f)
        depth_file = (depth_file * 256).astype(np.uint16)
        n_images = depth_file.shape[0]
        out_dir = join(sub_path, 'depth')
        create_dir(out_dir)
        for idx in range(n_images):
            Image.fromarray(depth_file[idx, ...]).save(join(out_dir, '%05d.png' % idx))
            
    if os.path.exists(sem_f):
        sem_file = np.load(sem_f)
        sem_file = sem_file.astype(np.uint8)
        n_images = sem_file.shape[0]
        out_dir = join(sub_path, 'semantic')
        create_dir(out_dir)
        for idx in range(n_images):
            Image.fromarray(sem_file[idx, ...]).save(join(out_dir, '%05d.png' % idx))