import os
import cv2
import time
import yaml
import shutil
import argparse
import numpy as np
from os.path import join
from PIL import Image, ImageDraw
import sys
sys.path.append('.')
from Snerf.src.vis import visualize_semantic
from count_bbox import add_bbox


parser = argparse.ArgumentParser()
parser.add_argument('--bg_name', type=str, required=True)
parser.add_argument('--wkdir', type=str, default="wkdir_0", required=True) 
args = parser.parse_args()

wkdir = args.wkdir

def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print("Create dir: %s" % dir)
    # else:
    #     f_list = os.listdir(dir)
    #     for f in f_list:
    #         if os.path.isdir(os.path.join(dir, f)):
    #             shutil.rmtree(os.path.join(dir, f))
    #         else:
    #             os.remove(join(dir, f))
    #     print("Remove all files in %s" % dir)
        
result_dir = join('./annotation', args.bg_name)
create_dir(result_dir)

result_dir = join(result_dir, str(int(time.time())))
create_dir(result_dir)

# camera parameters.
bg_dir = join(wkdir, 'raw_data', 'background', args.bg_name)
os.system("cp %s %s" % (join(bg_dir, 'raw_target_poses.npy'), join(result_dir, 'target_poses.npy')))
P = np.load(join(bg_dir, 'intrinsic.npy'))
np.save(join(result_dir, 'intrinsic.npy'), P)       
# os.system("cp %s %s" % (join(bg_dir, 'bev_results.npy'), join(result_dir, 'bev_results.npy')))  # Save the bev_results.


# copy the bg data.
# Image.
os.system("cp -r %s %s" % (join(bg_dir, "rgb"), join(result_dir, "image")))
# Semantic.
os.system("cp -r %s %s" % (join(bg_dir, "semantic"), join(result_dir, "semantic")))
# bbox.
os.makedirs(join(result_dir, "bbox"))
n_images = len(os.listdir(join(result_dir, "image")))
for ct in range(n_images):
    os.system("touch %s" % (join(result_dir, "bbox", "%05d.txt" % ct)))
# depth.
os.system("cp -r %s %s" % (join(bg_dir, "depth"), join(result_dir, "depth")))


# Insert raw bboxes.
result_path = result_dir
scene_idx = args.bg_name[-7:]
scene_path = join("./waymo_scenes", scene_idx)
# dataset_path = join('./Snerf','full_datasets','datasets',scene_idx)
dataset_path = join('./Snerf', 'mv_datasets', scene_idx)
add_bbox(scene_path, result_path, dataset_path)

print("\n\n\nCongratulations... You have measured to the end.")