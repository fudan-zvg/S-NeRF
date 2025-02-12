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
# from Snerf.src.vis import visualize_semantic
from annotate_code.visualize import visualize_semantic
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

# images.
from_dir = "./%s/out_dir_stage3/image" % wkdir
os.system("cp -r %s %s" % (from_dir, result_dir))
n_images = len(os.listdir(from_dir))

# semantics.
from_dir = "./%s/out_dir_stage1/semantic" % wkdir
os.system("cp -r %s %s" % (from_dir, result_dir))

# bbox. 
from_dir = "./%s/out_dir_stage1/bbox" % wkdir
os.system("cp -r %s %s" % (from_dir, result_dir))

# depth.
from_dir = "./%s/out_dir_stage1/depth" % wkdir
os.system("cp -r %s %s" % (from_dir, result_dir))

# camera parameters.
bg_dir = join(wkdir, 'raw_data', 'background', args.bg_name)
# c2w = np.load(join(bg_dir, 'target_poses.npy'))
raw_c2w = np.load(join(bg_dir, 'raw_target_poses.npy'))
valid_idx_list, invalid_idx_list = np.load(join(bg_dir, "valid_idx.npy")).tolist(), np.load(join(bg_dir, "invalid_idx.npy")).tolist()
# c2w = c2w[:n_images, ...]        # only store the valid data.
c2w = np.concatenate([raw_c2w[valid_idx_list], raw_c2w[invalid_idx_list]], axis=0) 
np.save(join(result_dir, 'target_poses.npy'), c2w)
P = np.load(join(bg_dir, 'intrinsic.npy'))
np.save(join(result_dir, 'intrinsic.npy'), P)       
os.system("cp %s %s" % (join(bg_dir, 'bev_results.npy'), join(result_dir, 'bev_results.npy')))  # Save the bev_results.

# copy the invalid data.
invalid_idx_list = np.load(join(bg_dir, "invalid_idx.npy")).tolist()
ct = n_images
for invalid_idx in invalid_idx_list:
    # Image.
    os.system("cp %s %s" % (join(bg_dir, "rgb", "%05d.png" % invalid_idx), join(result_dir, "image", "%05d.png" % ct)))
    # Semantic.
    os.system("cp %s %s" % (join(bg_dir, "semantic", "%05d.png" % invalid_idx), join(result_dir, "semantic", "%05d.png" % ct)))
    # bbox.
    os.system("touch %s" % (join(result_dir, "bbox", "%05d.txt" % ct)))
    # depth.
    os.system("cp %s %s" % (join(bg_dir, "depth", "%05d.png" % invalid_idx), join(result_dir, "depth", "%05d.png" % ct)))
    ct += 1

# Insert raw bboxes.
result_path = result_dir
scene_idx = args.bg_name[-7:]
# scene_path = join("./waymo_scenes", scene_idx)
scene_path = join('./dataset/raw_dataset', scene_idx)

# dataset_path = join('./Snerf','full_datasets','datasets',scene_idx)
# dataset_path = join('./Snerf', 'mv_datasets', scene_idx)
dataset_path = join('./dataset/processed_dataset', scene_idx)
add_bbox(scene_path, result_path, dataset_path)

print("\n\n\nCongratulations... You have measured to the end.")