import os
import cv2
import shutil
import argparse
import numpy as np
from PIL import Image

from annotation_utils import *

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
                os.remove(os.path.join(dir, f))
        print("Remove all files in %s" % dir)


def visualize_one_dir(in_dir):
    
    image_dir = os.path.join(in_dir, 'image')    
    semantic_dir = os.path.join(in_dir, 'semantic')
    n_images = len(os.listdir(image_dir))
    
    vis_dir = os.path.join(in_dir, 'visualize')
    create_dir(vis_dir)
    sem_vis_dir = os.path.join(vis_dir, 'semantic')
    bbox_vis_dir = os.path.join(vis_dir, 'bbox')
    bev_vis_dir = os.path.join(vis_dir, 'bev')
    create_dir(sem_vis_dir)
    create_dir(bbox_vis_dir)
    create_dir(bev_vis_dir)
    
    for idx in range(n_images):
        
        im_mat = np.array(Image.open(os.path.join(image_dir, "%05d.png" % idx)))
        # 1. semantic.
        sem_mat = np.array(Image.open(os.path.join(semantic_dir, "%05d.png" % idx)))
        sem_mat = (sem_mat / 20 * 255).astype(np.uint8)   # Normalize.
        sem_mat = cv2.cvtColor(sem_mat, cv2.COLOR_GRAY2RGB)   # gray -> rgb.
        im_sem_mat = cv2.addWeighted(sem_mat, .5, im_mat, .5, gamma=0)
        cv2.imwrite(os.path.join(sem_vis_dir, "%05d.png" % idx), im_sem_mat)
        # 2. bbox.
        vis_im, bev_im = visualize_one_rec(in_dir, idx)
        vis_im.save(os.path.join(bbox_vis_dir, "%05d.png" % idx))
        # 3. bev.
        bev_im.save(os.path.join(bev_vis_dir, "%05d.png" % idx))


from matplotlib import cm
import matplotlib.pyplot as plt


def visualize_depth(depth, pathname, idx):
    colormap = cm.get_cmap('turbo')
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = depth.min() + eps
    far = depth.max() - eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]
    os.makedirs('./test_depths/' + pathname, exist_ok=True)
    Image.fromarray(
        (np.clip(np.nan_to_num(vis), 0., 1.) * 255.).astype(np.uint8)).save(
        os.path.join('./test_depths/', pathname, 'depth_vis_{:04d}.png'.format(idx)))


def visualize_semantic(semantic, pathname, idx):
    plt.figure(figsize=(64, 60))
    plt.imshow(semantic)
    plt.grid(False)
    plt.axis('off')
    os.makedirs(os.path.join(pathname, 'test_semantics'), exist_ok=True)
    plt.savefig(os.path.join(pathname, 'test_semantics', 'semantic_vis_{:04d}.png'.format(idx)))
    plt.clf()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg_name", required=True, type=str)
    args = parser.parse_args()
    
    result_dir = os.path.join("annotation", args.bg_name)
    sub_dir_list = sorted(os.listdir(result_dir))
    
    for sub_dir in sub_dir_list:
        # if sub_dir == "1676449934":
        # if "bev_results.npy" in os.listdir(os.path.join(result_dir, sub_dir)):
        one_dir = os.path.join(result_dir, sub_dir)
        visualize_one_dir(one_dir)