import os
import cv2
import copy
from math import atan2, sqrt, sin, cos
from PIL import Image
import argparse
import trimesh
import numpy as np

from rasterizer import rasterize
from utils_render import get_mask_from_mesh_use_inv

def look_at_rotation(camera_position, up=((0, 0, 1),)):
    
    T = camera_position
    x, y, z = camera_position
    yaw_angle = atan2(y, x)
    pitch_angle = atan2(z, sqrt(y**2+x**2))
    # import pdb; pdb.set_trace()
    R = np.array([
        [cos(yaw_angle), -sin(yaw_angle), 0],
        [sin(yaw_angle), cos(yaw_angle), 0],
        [0, 0, 1]
    ])
    x_rot = np.array([
        [0, 0, -1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    z_rot = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    ####################### FIXME #######################
    pitch_rot = np.array([
        [1, 0, 0],
        [0, cos(pitch_angle), sin(pitch_angle)],
        [0, -sin(pitch_angle), cos(pitch_angle)]
    ])
    # pitch_rot = np.array([
    #     [cos(pitch_angle), 0, -sin(pitch_angle)],
    #     [0, 1, 0],
    #     [sin(pitch_angle), 0, cos(pitch_angle)]
    # ])
    # pitch_rot = np.eye(3)
    ####################### FIXME #######################
    
    return z_rot @ R @ x_rot @ pitch_rot


def process_v2c(all_v2c, is_look_at=False):
    for idx in range(all_v2c.shape[0]):
        all_v2c[idx] = np.linalg.inv(all_v2c[idx])
        if is_look_at:
            all_v2c[idx, :3, :3] = look_at_rotation(all_v2c[idx, :3, 3], up=((0, 0, 1),))
            all_v2c[idx, :3, :1] = all_v2c[idx, :3, :1] * -1.   # Not sure. In colmap.py, this step is moved outside this function. (But exists)
        else:
            all_v2c[idx, :3, :3] = -1 * all_v2c[idx, :3, :3]
        all_v2c[idx] = np.linalg.inv(all_v2c[idx])     
    return all_v2c  


def get_raw_mask(dir, idx):
    path = os.path.join(dir, "%06d.png" % idx)
    return np.array(Image.open(path))


def match(raw_mask, mask, category="vehicle"):
    
    mask_sel = raw_mask[..., 0]
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    area = []
    counter_area = []

    if len(contours) == 0:
        return mask

    mask_sel = np.ascontiguousarray(mask_sel)
    for k in range(len(contours)):
        temp_mask = mask_sel.copy()
        temp_mask = np.ascontiguousarray(temp_mask)
        for i in range(len(contours)):
            if i != k:
                cv2.fillPoly(temp_mask, [contours[i]], 0)
        area.append((temp_mask.astype(bool) & mask[..., 0].astype(bool)).sum())
        counter_area.append(temp_mask.astype(bool).sum())
    
    max_idx = np.argmax(area)
    max_area = area[max_idx]
    max_counter_area = counter_area[max_idx]
    mask_area = mask[..., 0].astype(bool).sum()

    area_ratio_thres = 1.5 if category == "vehicle" else 3
    if max_counter_area / mask_area > area_ratio_thres:
        if category == "vehicle":
            print("Match fail, use raw mask... (ratio: %f)" % (max_counter_area / max_area))  
            return mask
        else:
            print("Match fail, use raw mask...")
            return mask
    
    for k in range(len(contours)):
        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)    
            
    mask_sel = np.expand_dims(mask_sel, axis=2).repeat(3, axis=2)
    return mask_sel


def get_masks(args):
    
    all_c2v = np.load(args.calib_dir)        # [N, 4, 4]
    all_v2c = np.linalg.inv(all_c2v)
    P = np.load(args.intrinsic_dir)[0]       # [4, 4]
    raw_mesh = trimesh.load(args.mesh_dir)
    raw_mask_dir = args.raw_mask_dir
    image_dir = args.image_dir
    resolution = (args.W, args.H)
    save_dir = args.save_dir
    mask_dir = os.path.join(save_dir, "mask")
    mesh_mask_dir = os.path.join(save_dir, "mesh_mask")
    depth_dir = os.path.join(save_dir, "depth")
    category = args.category
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    if not os.path.exists(mesh_mask_dir):
        os.makedirs(mesh_mask_dir)
    # if not os.path.exists(depth_dir):
    #     os.makedirs(depth_dir)        
    
    all_v2c = process_v2c(all_v2c, is_look_at=True)
    
    for idx, v2c in enumerate(all_v2c):
        _mesh = copy.deepcopy(raw_mesh)
        mesh = _mesh.apply_transform(v2c)
        mesh_mask = get_mask_from_mesh_use_inv(mesh, resolution, P)
        Image.fromarray(mesh_mask).save(os.path.join(mesh_mask_dir, "%06d.png" % idx))
        raw_mask = get_raw_mask(raw_mask_dir, idx)
        mask = match(raw_mask, mesh_mask, category)
        Image.fromarray(mask).save(os.path.join(mask_dir, "%06d.png" % idx))

        
        
        
if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib_dir', type=str)
    parser.add_argument('--intrinsic_dir', type=str)
    parser.add_argument('--mesh_dir', type=str)
    parser.add_argument('--raw_mask_dir', type=str)
    parser.add_argument('--W', type=int)
    parser.add_argument('--H', type=int)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--category', type=str, default="vehicle")
    args = parser.parse_args()
    
    assert len(list(filter(os.path.exists, [args.calib_dir, args.intrinsic_dir, args.mesh_dir]))) == 3, "invalid dirs."
    
    get_masks(args)