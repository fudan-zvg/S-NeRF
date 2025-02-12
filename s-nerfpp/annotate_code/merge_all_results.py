import os
import time
import yaml
import shutil
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


result_dir = join('./annotation')

sub_result_dirs = [sub_dir for sub_dir in sorted(os.listdir(result_dir)) if sub_dir != 'final_results' and sub_dir != 'README.txt']
out_dir = join(result_dir, 'final_results')
create_dir(out_dir)

# images.
image_dir = join(out_dir, 'image')
create_dir(image_dir)

ct = 0
for sub_dir in sub_result_dirs:
    sub_path = join(result_dir, sub_dir)
    sub_sub_result_dirs = [_ for _ in sorted(os.listdir(sub_path)) if _ != 'merge_results']
    for sub_sub_dir in sub_sub_result_dirs:
        sub_sub_path = os.path.join(sub_path, sub_sub_dir, 'image')
        for im_fname in sorted(os.listdir(sub_sub_path)):
            from_path = join(sub_sub_path, im_fname)
            to_path = join(image_dir, '%05d.png' % ct)
            os.system('cp %s %s' % (from_path, to_path))
            ct += 1
        print("Finish! %s" % sub_sub_path)
        print("ct for %s: %d" % (sub_sub_path, ct))

# semantics.
semantic_dir = join(out_dir, 'semantic')
create_dir(semantic_dir)

ct = 0
for sub_dir in sub_result_dirs:
    sub_path = join(result_dir, sub_dir)
    sub_sub_result_dirs = [_ for _ in sorted(os.listdir(sub_path)) if _ != 'merge_results']
    for sub_sub_dir in sub_sub_result_dirs:
        sub_sub_path = os.path.join(sub_path, sub_sub_dir, 'semantic')
        for semantic_fname in sorted(os.listdir(sub_sub_path)):
            from_path = join(sub_sub_path, semantic_fname)
            to_path = join(semantic_dir, '%05d.png' % ct)
            os.system('cp %s %s' % (from_path, to_path))
            ct += 1
        print("Finish! %s" % sub_sub_path)    
        print("ct for %s: %d" % (sub_sub_path, ct))    
        
# bbox.
bbox_dir = join(out_dir, 'bbox')
create_dir(bbox_dir)

ct = 0
for sub_dir in sub_result_dirs:
    sub_path = join(result_dir, sub_dir)
    sub_sub_result_dirs = [_ for _ in sorted(os.listdir(sub_path)) if _ != 'merge_results']
    for sub_sub_dir in sub_sub_result_dirs:
        sub_sub_path = os.path.join(sub_path, sub_sub_dir, 'bbox')
        for bbox_fname in sorted(os.listdir(sub_sub_path)):
            from_path = join(sub_sub_path, bbox_fname)
            to_path = join(bbox_dir, '%05d.txt' % ct)
            os.system('cp %s %s' % (from_path, to_path))
            ct += 1
        print("Finish! %s" % sub_sub_path) 
        print("ct for %s: %d" % (sub_sub_path, ct))         
        
# Camera parameters.
P_list = []; c2w_list = []
for sub_dir in sub_result_dirs:
    sub_path = join(result_dir, sub_dir)
    sub_sub_result_dirs = [_ for _ in sorted(os.listdir(sub_path)) if _ != 'merge_results']
    for sub_sub_dir in sub_sub_result_dirs:
        P_list.append(np.load(join(sub_path, sub_sub_dir, 'intrinsic.npy')))
        c2w_list.append(np.load(join(sub_path, sub_sub_dir, 'target_poses.npy')))
Ps = np.stack(P_list, axis=0); c2ws = np.concatenate(c2w_list, axis=0)
np.save(join(out_dir, 'intrinsic.npy'), Ps)
np.save(join(out_dir, 'target_poses.npy'), c2ws)