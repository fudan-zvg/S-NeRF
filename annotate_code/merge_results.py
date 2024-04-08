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


parser = argparse.ArgumentParser()
parser.add_argument('--bg_name', type=str, required=True)
args = parser.parse_args()

result_dir = join('./annotation', args.bg_name)

sub_result_dirs = [sub_dir for sub_dir in sorted(os.listdir(result_dir)) if sub_dir != 'merge_results']
out_dir = join(result_dir, 'merge_results')
create_dir(out_dir)

# images.
image_dir = join(out_dir, 'image')
create_dir(image_dir)

ct = 0
for sub_dir in sub_result_dirs:
    sub_path = join(result_dir, sub_dir, 'image')
    for im_fname in sorted(os.listdir(sub_path)):
        from_path = join(sub_path, im_fname)
        to_path = join(image_dir, '%05d.png' % ct)
        os.system('cp %s %s' % (from_path, to_path))
        ct += 1

# semantics.
semantic_dir = join(out_dir, 'semantic')
create_dir(semantic_dir)

ct = 0
for sub_dir in sub_result_dirs:
    sub_path = join(result_dir, sub_dir, 'semantic')
    for semantic_fname in sorted(os.listdir(sub_path)):
        from_path = join(sub_path, semantic_fname)
        to_path = join(semantic_dir, '%05d.png' % ct)
        os.system('cp %s %s' % (from_path, to_path))
        ct += 1
        
# bbox.
bbox_dir = join(out_dir, 'bbox')
create_dir(bbox_dir)

ct = 0
for sub_dir in sub_result_dirs:
    sub_path = join(result_dir, sub_dir, 'bbox')
    for bbox_fname in sorted(os.listdir(sub_path)):
        from_path = join(sub_path, bbox_fname)
        to_path = join(bbox_dir, '%05d.txt' % ct)
        os.system('cp %s %s' % (from_path, to_path))
        ct += 1
        
# camera parameters.
P_list = []; c2w_list = []
for sub_dir in sub_result_dirs:
    P_list.append(np.load(join(result_dir, sub_dir, 'intrinsic.npy')))
    c2w_list.append(np.load(join(result_dir, sub_dir, 'target_poses.npy')))
Ps = np.concatenate(P_list, axis=0); c2ws = np.concatenate(c2w_list, axis=0)
np.save(join(out_dir, 'intrinsic.npy'), Ps)
np.save(join(out_dir, 'target_poses.npy'), c2ws)