import math
from pickletools import uint8
import cv2
import time
import random
import argparse
import sys
sys.path.append('../')
import raytracing
import numpy as np
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
from os.path import join

from scipy.spatial.transform import Rotation as R
import torch

# from ip_utils import get_bound, trans, scale

#### Random utils. ####
def random_noise(x, dx):
    return random.uniform(x-dx, x+dx)


def random_interval(interval):
    a, b = interval
    a, b = min(a, b), max(a, b)
    return random.uniform(a, b)

#### Read the data. #### 
def get_matrix(translation,
               rotation):
    '''
    Return a transformation matrix.
    '''
    rot = Quaternion(rotation).rotation_matrix
    trans = np.array(translation).reshape((3, 1))
    
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3:4] = trans
    
    return mat


def get_calib(idx, bg_dir):
    calibs = np.load(join(bg_dir, 'target_poses.npy'))
    intrinsics = np.load(join(bg_dir, 'intrinsic.npy'))
    if len(intrinsics.shape) == 2:
        return np.linalg.inv(calibs[idx, ...]), intrinsics
    return np.linalg.inv(calibs[idx, ...]), intrinsics[idx, ...]


def get_depth(idx, bg_dir):
    depth_path = join(bg_dir, 'depth', '%05d.png' % idx)
    depth_mat = np.array(Image.open(depth_path))
    depth_mat = depth_mat / 256.
    return depth_mat
    return np_file[idx, ...]


def get_semantic(idx, bg_dir, valid_HW=(1920, 1280)):
    sem_path = join(bg_dir, 'semantic', '%05d.png' % idx)
    sem_mat = np.array(Image.open(sem_path))
    
    # Only use valid HW.
    if sem_mat.shape[:2] != valid_HW:
        return None
        
    return sem_mat
    return np_file[idx, ...]


def get_semantic_points(bg_dir,
                        drop_ratio=.99,
                        max_depth=60.,
                        valid_HW=(1280, 1920)):
    '''
    Use projection to get points.
    '''

    # Load the data.
    calibs = torch.tensor(np.load(join(bg_dir, 'raw_target_poses.npy'))).float().cuda()
    intrinsic = np.load(join(bg_dir, 'intrinsic.npy'))
    intrinsics = np.expand_dims(intrinsic, axis=0).repeat(calibs.shape[0], axis=0)
    intrinsics = torch.tensor(intrinsics).float().cuda()
    # depths = torch.tensor(np.load(join(bg_dir, 'depth.npy'))).float().cuda()
    # semantics = torch.tensor(np.load(join(bg_dir, 'semantic.npy'))).int().cuda()
    
    n_images = calibs.shape[0]
    
    # Get the valid points. (We choose the valid points.)
    total_p = []
    total_semantic = []
    for idx in range(n_images):
        depth = torch.tensor(get_depth(idx, bg_dir)).float().cuda()
        semantic = get_semantic(idx, bg_dir, valid_HW=valid_HW)
        if semantic is None:
            continue
        semantic = torch.tensor(semantic).int().cuda()
        H, W = depth.shape[0], depth.shape[1]
        tx = torch.linspace(0, W - 1, W)
        ty = torch.linspace(0, H - 1, H)
        # pixels_x, pixels_y = torch.meshgrid(tx, ty)      
        # pixels_x, pixels_y = torch.flatten(pixels_x), torch.flatten(pixels_y)
        pixels_y, pixels_x = torch.where(depth.detach().cpu() < max_depth)
        
        # Choose points.
        n_pixels = pixels_x.shape[0]
        new_n_pixels = int(n_pixels * (1-drop_ratio))
        pt_index = torch.LongTensor(random.sample(range(n_pixels), new_n_pixels))
        pixels_x, pixels_y = torch.index_select(pixels_x, 0, pt_index), torch.index_select(pixels_y, 0, pt_index)
        # import pdb;pdb.set_trace()
        depth_val = depth[pixels_y.long(), pixels_x.long()]
        semantic_val = semantic[pixels_y.long(), pixels_x.long()]
        
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=0)  # shape: [3, N]
        p = depth_val * p.cuda()
        p = torch.matmul(torch.inverse(intrinsics[idx, ...]), p)
        p = torch.stack([p[0, :], p[1, :], p[2, :], torch.ones_like(p[0, :])], dim=0)  # shape: [4, N]
        p = torch.matmul(calibs[idx, ...], p)
        
        p = p.detach().cpu().numpy().T[:, :3]    # shape: [N, 4]
        semantic_val = semantic_val.detach().cpu().numpy()
        # p[:, 3] = semantic_val
        
        total_semantic.append(semantic_val)
        total_p.append(p)
    
    return np.concatenate(total_p, axis=0), np.concatenate(total_semantic, axis=0)
    


#### Get the camcalib. #### 

def get_camcalib(bg_dir,
                 idx,
                 target_coord,
                 base_angle,
                 mode="SNeRF"):
    '''
    Get the camera calibs.
    We compute the c2w (or c2v; w: The vehicle's coordinates).
    '''
    
    ##### Vehicle's coord -> World's coord.
    v2w = np.eye(4)
    
    # First, change the axis to compute.
    if mode == "SNeRF":
        axis_change = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        v2w = axis_change @ v2w
    
    # Then, rotate it.
    axis_rot = np.eye(4)
    rot = R.from_euler('z', base_angle, degrees=True).as_matrix()
    axis_rot[:3, :3] = rot
    v2w = axis_rot @ v2w
    
    # Finally, translation.
    axis_trans = np.eye(4)
    axis_trans[:3, 3:4] = np.array(target_coord).reshape((3,1))
    v2w = axis_trans @ v2w
    
    ##### World's coord -> Camera's coord.
    w2c, P = get_calib(idx, bg_dir)                      
    
    ##### Vehicle's coord -> World's coord -> Camera's coord.
    v2c = w2c @ v2w
    c2v = np.linalg.inv(v2c)
    
    # # Finally, change the axis back.
    # c2v = axis_change.T @ c2v
    
    return c2v, P, np.linalg.inv(w2c)
    
    
#### To get the largest counter region. ####
#### Refer to https://blog.csdn.net/qq_33854260/article/details/106297999 ####
def find_max_region(mask_sel):
    contours, hierarchy = cv2.findContours(mask_sel,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  
    area = []

    if len(contours) == 0:
        return mask_sel
    
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
 
    max_idx = np.argmax(area)
 
    max_area = cv2.contourArea(contours[max_idx])
 
    mask_sel = np.ascontiguousarray(mask_sel)
    for k in range(len(contours)):
    
        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    
    return mask_sel

    
#### To get the regions for placing vehicles and cameras. ####
def get_drivable_regions(points,
                         semantics,
                         drivable_semantic_idx,
                         undrivable_semantic_idx_list,
                         obstacle_semantic_idx_list):
    '''
    semantic_points: [N, 4].
    '''
    
    def build_bev(xy_points,
                  undrivable_xy_points,
                  obstacle_xy_points):
        '''
        xy_points: [N, 2]
        bev_result: 
        {
            bev_map: [bev_H, bev_W],
            x_ori_mean, y_ori_mean: float
            x_scale, y_scale: float.    
        }
        '''
        
        x_points, y_points, height = xy_points[:, 0], xy_points[:, 1], xy_points[:, 2]
        mean_drivable_height = height.mean()        
        if undrivable_xy_points is not None:
            mask = np.ones_like(undrivable_xy_points[:, 2])
            mask = np.logical_and(undrivable_xy_points[:, 2] < (mean_drivable_height + 5), mask)
            undrivable_xy_points = undrivable_xy_points[mask, :]
            undrivable_x_points, undrivable_y_points, undrivable_height = undrivable_xy_points[:, 0], undrivable_xy_points[:, 1], undrivable_xy_points[:, 2]
        if obstacle_xy_points is not None:
            obstacle_x_points, obstacle_y_points, obstacle_height = obstacle_xy_points[:, 0], obstacle_xy_points[:, 1], obstacle_xy_points[:, 2]            
        x_range, y_range = x_points.max() - x_points.min(), y_points.max() - y_points.min()
        
        bev_W, bev_H = int(x_range * 10), int(y_range * 10)                 # the grid size is almost 0.1.
        bev_map = np.zeros((bev_H, bev_W), dtype=np.uint8)
        height_map = np.zeros((bev_H, bev_W), dtype=np.float32)
        undrivable_map = np.zeros_like(bev_map)
        obstacle_map = np.zeros_like(bev_map)
        
        # Recenter the points.
        x_ori_bias = x_points.min()
        y_ori_bias = y_points.min()
        x_points -= x_ori_bias
        y_points -= y_ori_bias
        if undrivable_xy_points is not None:
            undrivable_x_points -= x_ori_bias
            undrivable_y_points -= y_ori_bias
        if obstacle_xy_points is not None:
            obstacle_x_points -= x_ori_bias
            obstacle_y_points -= y_ori_bias
        
        # Scale
        x_scale, y_scale = .95 * bev_W / x_range, .95 * bev_H / y_range
        x_points, y_points = x_points * x_scale, y_points * y_scale
        if undrivable_xy_points is not None:
            undrivable_x_points, undrivable_y_points = undrivable_x_points * x_scale, undrivable_y_points * y_scale
        if obstacle_xy_points is not None:
            obstacle_x_points, obstacle_y_points = obstacle_x_points * x_scale, obstacle_y_points * y_scale
        
        # Record to the bev.
        x_points, y_points = x_points.astype(np.uint16), y_points.astype(np.uint16)
        if undrivable_xy_points is not None:
            undrivable_x_points, undrivable_y_points = undrivable_x_points.astype(np.uint16), undrivable_y_points.astype(np.uint16)
            # Clip the outside points.
            mask = np.ones_like(undrivable_x_points, dtype=bool)
            mask = np.logical_and(mask, undrivable_x_points > 0)
            mask = np.logical_and(mask, undrivable_x_points < bev_W)
            mask = np.logical_and(mask, undrivable_y_points > 0)
            mask = np.logical_and(mask, undrivable_y_points < bev_H)
            undrivable_x_points, undrivable_y_points = undrivable_x_points[mask], undrivable_y_points[mask]
            undrivable_height = undrivable_height[mask]
        if obstacle_xy_points is not None:
            obstacle_x_points, obstacle_y_points = obstacle_x_points.astype(np.uint16), obstacle_y_points.astype(np.uint16)
            mask = np.ones_like(obstacle_x_points, dtype=bool)
            mask = np.logical_and(mask, obstacle_x_points > 0)
            mask = np.logical_and(mask, obstacle_x_points < bev_W)
            mask = np.logical_and(mask, obstacle_y_points > 0)
            mask = np.logical_and(mask, obstacle_y_points < bev_H)
            obstacle_x_points, obstacle_y_points = obstacle_x_points[mask], obstacle_y_points[mask]
            obstacle_height = obstacle_height[mask]            
        
        bev_map[y_points, x_points] = 255        # Record the drivable regions.
        height_map[y_points, x_points] = height
        if undrivable_xy_points is not None:
            undrivable_map[undrivable_y_points, undrivable_x_points] = 255   # Record the undrivable regions.
        if obstacle_xy_points is not None:
            obstacle_map[obstacle_y_points, obstacle_x_points] = 255         # Record the obstacles.
        
        # Now handle the bev map. 
        r = max(bev_H, bev_W) // 300
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r))
        smaller_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(r//5,1), max(r//5,1)))
        bev_map_refined = cv2.morphologyEx(bev_map, cv2.MORPH_CLOSE, kernel, iterations=1)
        if undrivable_xy_points is not None:
            undrivable_map = cv2.morphologyEx(undrivable_map, cv2.MORPH_OPEN, smaller_kernel, iterations=1)
        if obstacle_xy_points is not None:
            obstacle_map = cv2.morphologyEx(obstacle_map, cv2.MORPH_OPEN, smaller_kernel, iterations=1)
        
        # # Erode to avoid something bad.   1 meter to erode.
        # erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        # bev_map_refined = cv2.erode(bev_map, erode_kernel, iterations=1)
        
        if undrivable_xy_points is not None:
            bev_map_refined[undrivable_map > 0] = 127     # Record the undrivable regions.
        if obstacle_xy_points is not None:
            bev_map_refined[obstacle_map > 0] = 64        # Record the obstacles.
        
        # cv2.imwrite("/var/lib/docker/data/users/liwenye/bev_map.png", bev_map)
        # cv2.imwrite("/var/lib/docker/data/users/liwenye/bev_map_refined.png", bev_map_refined)
        # import pdb; pdb.set_trace()
        
        return {
            "x_ori_bias": x_ori_bias,
            "y_ori_bias": y_ori_bias,
            "x_scale": x_scale,
            "y_scale": y_scale,
            "bev_map": bev_map,
            "bev_map_refined": bev_map_refined,
            "height_map": height_map
        }
        
    
    # drivable_points = semantic_points[semantic_points[:, 3] == drivable_semantic_idx]
    drivable_points = points[semantics==drivable_semantic_idx, :]
    
    if undrivable_semantic_idx_list is not None:
        undrivable_points_list = []
        for undrivable_semantic_idx in undrivable_semantic_idx_list:
            undrivable_points_list.append(points[semantics==undrivable_semantic_idx, :])
        undrivable_points = np.concatenate(undrivable_points_list, axis=0)
    else:
        undrivable_points = None
        
    if obstacle_semantic_idx_list is not None:
        obstacle_points_list = []
        for obstacle_semantic_idx in obstacle_semantic_idx_list:
            obstacle_points_list.append(points[semantics==obstacle_semantic_idx, :])
        obstacle_points = np.concatenate(obstacle_points_list, axis=0)
    else:
        obstacle_points = None
    
    # First get the height.
    # ground_height = drivable_points[:, 2].mean()
    
    # Now we find the drivable regions.
    # drivable_points = drivable_points[:, :2]   # Only x and y.
    
    # Build the bev.
    print("Start building bev!")
    bev_result = build_bev(drivable_points, undrivable_points, obstacle_points)
    print("Finish building bev!")
    return bev_result


def sample_pos_from_bev(bev_result, reject_r, render_pose, intrinsic):
    '''
    reject_r: The minimum distance between vehicle and undrivable regions.
    '''
    
    def mask_invisible_regions(bev_map, render_pose, intrinsic):
        
        def inv_proj(x, y):
            ray_point_2D = np.array([x, y, 1]).reshape((3,1))   # We set the depth as 1.
            ray_point_3D = np.linalg.inv(intrinsic) @ ray_point_2D
            ray_point_3D = np.concatenate((ray_point_3D, [[1,]]), axis=0)
            ray_point_3D = render_pose @ ray_point_3D
            ray_point_3D = ray_point_3D[:3, 0]      
            return ray_point_3D      
        
        # First get the ray_o and ray_d.
        H, W = (2 * intrinsic[1, 2]), (2 * intrinsic[0, 2])
        focal = intrinsic[0, 0]      # Assume focal_x == focal_y.
        ray_point_3D = inv_proj(W/2, H/2)
        ray_o = render_pose[:3, 3]        
        ray_bev_o = np.array([(ray_o[0] - x_ori_bias) * x_scale, (ray_o[1] - y_ori_bias) * y_scale])
        ray_bev_point = np.array([(ray_point_3D[0] - x_ori_bias) * x_scale, (ray_point_3D[1] - y_ori_bias) * y_scale])
        ray_bev_d = ray_bev_point - ray_bev_o
        ray_bev_d = ray_bev_d / np.linalg.norm(ray_bev_d)
        
        # Then we calculate the fov.
        edge_point_3D = inv_proj(W, H/2)
        edge_bev_point = np.array([(edge_point_3D[0] - x_ori_bias) * x_scale, (edge_point_3D[1] - y_ori_bias) * y_scale])
        edge_bev_d = edge_bev_point - ray_bev_o
        edge_bev_d = edge_bev_d / np.linalg.norm(edge_bev_d)
        cos_thres = np.dot(edge_bev_d, ray_bev_d)
        assert cos_thres > 0
        
        # Finally we calculate all cos to filter regions.
        bev_H, bev_W = bev_map.shape
        tx = torch.linspace(0, bev_W - 1, bev_W)
        ty = torch.linspace(0, bev_H - 1, bev_H)
        pos_x, pos_y = torch.meshgrid(tx, ty)      
        pos_x, pos_y = torch.flatten(pos_x), torch.flatten(pos_y)        
        vec_x, vec_y = pos_x - ray_bev_o[0], pos_y - ray_bev_o[1]     # Vector.
        vecs = torch.stack((vec_x, vec_y), dim=0).cuda()   # 2 * N
        vecs = vecs / torch.norm(vecs, dim=0)
        ray_bev_d = torch.tensor(ray_bev_d).float().unsqueeze(dim=0).cuda()
        cos_vals = torch.matmul(ray_bev_d, vecs).detach().cpu().squeeze(dim=0)
        invalid_pos_x, invalid_pos_y = pos_x[cos_vals < cos_thres], pos_y[cos_vals < cos_thres]
        invalid_pos_x, invalid_pos_y = invalid_pos_x.numpy().astype(np.int32), invalid_pos_y.numpy().astype(np.int32)
        
        masked_bev_map = bev_map.copy()
        masked_bev_map[invalid_pos_y, invalid_pos_x] = 0
        return masked_bev_map
    
    ori_bev_map = bev_result["bev_map"]
    bev_map = bev_result["bev_map_refined"]
    height_map = bev_result["height_map"]
    bev_H, bev_W = bev_map.shape
    x_scale, y_scale = bev_result["x_scale"], bev_result["y_scale"]
    x_ori_bias, y_ori_bias = bev_result["x_ori_bias"], bev_result["y_ori_bias"]
    
    masked_bev_map = mask_invisible_regions(bev_map, render_pose, intrinsic)
    # y_list, x_list = np.where(bev_map > 0)
    y_list, x_list = np.where(masked_bev_map > 0)
    n_total_pos = y_list.size
    # A while loop to sample.
    ct = 0
    fail_flag = False
    
    if n_total_pos == 0:
        fail_flag = True
        return 0, 0, 0, fail_flag
    
    while True:
        idx = np.random.choice(list(range(n_total_pos)), size=1)[0]
        x_bev, y_bev = x_list[idx], y_list[idx]
        
        # Judge the surrounding.
        reject_pixel_x, reject_pixel_y = int(reject_r * x_scale), int(reject_r * y_scale)
        surrounding_region = bev_map[max(0, y_bev-reject_pixel_y) : min(bev_H, y_bev+reject_pixel_y+1), \
                                    max(0, x_bev-reject_pixel_x) : min(bev_W, x_bev+reject_pixel_x+1)]
        undrivable_area = (surrounding_region == 127).sum()
        obstacle_area = (surrounding_region == 64).sum()
        drivable_area = (surrounding_region == 255).sum()
        if ct > 20:
            fail_flag = True
            break
        
        elif drivable_area == 0 or undrivable_area / drivable_area > 1 or obstacle_area / drivable_area > .2:
            ct += 1
            continue
        
        else:
            break
    
    # Get the ground height.
    ego_region = ori_bev_map[max(y_bev-bev_H//20, 0) : min(y_bev+bev_H//20, bev_H-1), \
                             max(x_bev-bev_W//20, 0) : min(x_bev+bev_W//20, bev_W-1)]
    ego_region_height = height_map[max(y_bev-bev_H//20, 0) : min(y_bev+bev_H//20, bev_H-1), \
                                   max(x_bev-bev_W//20, 0) : min(x_bev+bev_W//20, bev_W-1)]
    height_data = ego_region_height[ego_region > 0]
    
    if height_data.size == 0:
        fail_flag = True
        z_world = -9999
    else:
        z_world = height_data.mean()
    
    # From bev to world space.
    x_bev, y_bev = x_bev / x_scale, y_bev / y_scale
    x_world, y_world = x_bev + x_ori_bias, y_bev + y_ori_bias
    
    return x_world, y_world, z_world, fail_flag


def generate_pos_from_render_poses(render_poses,
                                   intrinsic,
                                   points,
                                   semantics,
                                   drivable_semantic_idx,
                                   undrivable_semantic_idx_list,
                                   obstacle_semantic_idx_list,
                                   instances_n,
                                   depth_range=(20, 60),
                                   reject_r=5,
                                   min_dist=15,
                                   no_sample_idx=None,):
    '''
    Generate the positions for vehicles according to bg_pose.
        min_dist: The minimum distance among vehicles.
        depth_range: The vehicle should be placed.
        reject_r: The minimum distance between vehicle and undrivable regions.
        no_sample_idx: The idx which not need to be sampled.
    '''
    
    def dist(pos1, pos2):
        '''
        pos: [x, y, z].
        '''
        return np.linalg.norm(np.array(pos1) - np.array(pos2))    
    
    def vehicle_in_camera_view(one_pos, render_pose, intrinsic):
        one_pos = np.array(one_pos).reshape((3,1))
        H, W = intrinsic[1, 2] * 2, intrinsic[0, 2] * 2
        
        # Proj.
        one_pos = np.concatenate([one_pos, [[1]]], axis=0)
        cam_coord = np.linalg.inv(render_pose) @ one_pos    # c2w -> w2c.
        if cam_coord[2, 0] < depth_range[0] or cam_coord[2, 0] > depth_range[1]:
            return False
        plane_coords = intrinsic @ cam_coord[:3, ...]
        plane_coords = plane_coords / plane_coords[2, 0]
        x, y = plane_coords[0, 0], plane_coords[1, 0]
        if x < 0 or x > W or y < 0 or y > H:
            return False
        
        return True
    

    n_images = render_poses.shape[0]
    
    # Get the bev map.
    bev_result = get_drivable_regions(points,
                                      semantics,
                                      drivable_semantic_idx,
                                      undrivable_semantic_idx_list,
                                      obstacle_semantic_idx_list)
    world_coord_list_vehicles, base_angle_list_vehicles =  [[] for _ in range(instances_n)], [[] for _ in range(instances_n)]
    valid_idx = []   
    invalid_idx = []                 
    for idx in range(n_images):
        render_pose = render_poses[idx]
        out_ct = 0
        final_fail_flag = False
        while True:
            fail_flag = False
            temp = []
            for instance_idx in range(instances_n):
                ct = 0
                while True:
                    reject_flag = False
                    x, y, z, reject_flag = sample_pos_from_bev(bev_result, reject_r, render_pose, intrinsic)
                    one_pos = [x, y, z]
                    for already_pos in temp:
                        if dist(already_pos, one_pos) < min_dist:   # To avoid the overlap of vehicles.
                            reject_flag = True                      # Generate a new pos.                
                    if not vehicle_in_camera_view(one_pos.copy(), render_pose, intrinsic):     # To judge whether the vehicle in the camera view.
                        reject_flag = True   
                    
                    ct += 1
                    
                    if ct > 10 or (no_sample_idx is not None and idx in no_sample_idx):     # Not able to get the valid idx or no need to sample.
                        fail_flag = True
                        break                
                    
                    if reject_flag:
                        continue          # Resample.
                    else:
                        break
                    
                temp.append(one_pos)
                # world_coord_list_vehicles[instance_idx].append(one_pos)
                # base_angle_list_vehicles[instance_idx].append(random_interval([0, 360]))      # FIXME!
                # raise NotImplementedError
            
            out_ct += 1
            
            if out_ct > 10 or (no_sample_idx is not None and idx in no_sample_idx):         # Not able to get the valid idx or no need to sample.
                if out_ct > 10:
                    print("Warning! We have droped an invalid idx: %d" % idx)
                else:
                    print("We skip invalid idx :%d" % idx)
                final_fail_flag = True
                break
                
            if fail_flag:
                continue
            
            else:
                print("Have tried %d times. idx: %d" % (out_ct, idx))
                break
            
        for instance_idx in range(instances_n):
            world_coord_list_vehicles[instance_idx].append(temp[instance_idx])
            base_angle_list_vehicles[instance_idx].append(random_interval([0, 360]))        
        
        if not final_fail_flag:            # Mean the index is valid.
            valid_idx.append(idx)                             
        else:
            invalid_idx.append(idx)

    # import pdb; pdb.set_trace()
    new_render_poses = render_poses[valid_idx, ...]
    new_world_coord_list_vehicles = [np.array(_)[valid_idx].tolist() for _ in world_coord_list_vehicles]
    new_base_angle_list_vehicles = [np.array(_)[valid_idx].tolist() for _ in base_angle_list_vehicles]
    new_n_images = len(valid_idx)
    
    return new_world_coord_list_vehicles, new_base_angle_list_vehicles, new_render_poses, new_n_images, bev_result, valid_idx, invalid_idx


def demo_poses_generator(render_poses,
                                   intrinsic,
                                   points,
                                   semantics,
                                   drivable_semantic_idx,
                                   undrivable_semantic_idx_list,
                                   obstacle_semantic_idx_list,
                                   instances_n,
                                   depth_range=(20, 60),
                                   reject_r=5,
                                   min_dist=15,
                                   no_sample_idx=None):
    '''
    Generate the positions for vehicles according to bg_pose.
        min_dist: The minimum distance among vehicles.
        depth_range: The vehicle should be placed.
        reject_r: The minimum distance between vehicle and undrivable regions.
        no_sample_idx: The idx which not need to be sampled.
    '''

    def dist(pos1, pos2):
        '''
        pos: [x, y, z].
        '''
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def vehicle_in_camera_view(one_pos, render_pose, intrinsic):
        one_pos = np.array(one_pos).reshape((3, 1))
        H, W = intrinsic[1, 2] * 2, intrinsic[0, 2] * 2

        # Proj.
        one_pos = np.concatenate([one_pos, [[1]]], axis=0)
        cam_coord = np.linalg.inv(render_pose) @ one_pos  # c2w -> w2c.
        if cam_coord[2, 0] < depth_range[0] or cam_coord[2, 0] > depth_range[1]:
            return False
        plane_coords = intrinsic @ cam_coord[:3, ...]
        plane_coords = plane_coords / plane_coords[2, 0]
        x, y = plane_coords[0, 0], plane_coords[1, 0]
        if x < 0 or x > W or y < 0 or y > H:
            return False

        return True

    n_images = render_poses.shape[0]

    # Get the bev map.
    bev_result = get_drivable_regions(points,
                                      semantics,
                                      drivable_semantic_idx,
                                      undrivable_semantic_idx_list,
                                      obstacle_semantic_idx_list)
    world_coord_list_vehicles, base_angle_list_vehicles = [[] for _ in range(instances_n)], [[] for _ in
                                                                                             range(instances_n)]
    valid_idx = []
    invalid_idx = []
    for idx in range(1):
        render_pose = render_poses[idx*(n_images-1)]
        out_ct = 0
        final_fail_flag = False
        while True:
            fail_flag = False
            temp = []
            for instance_idx in range(instances_n):
                ct = 0
                while True:
                    reject_flag = False
                    x, y, z, reject_flag = sample_pos_from_bev(bev_result, reject_r, render_pose, intrinsic)
                    one_pos = [x, y, z]
                    for already_pos in temp:
                        if dist(already_pos, one_pos) < min_dist:  # To avoid the overlap of vehicles.
                            reject_flag = True  # Generate a new pos.
                    if not vehicle_in_camera_view(one_pos.copy(), render_pose,
                                                  intrinsic):  # To judge whether the vehicle in the camera view.
                        reject_flag = True

                    ct += 1

                    if ct > 10 or (
                            no_sample_idx is not None and idx in no_sample_idx):  # Not able to get the valid idx or no need to sample.
                        fail_flag = True
                        break

                    if reject_flag:
                        continue  # Resample.
                    else:
                        break

                temp.append(one_pos)
                # world_coord_list_vehicles[instance_idx].append(one_pos)
                # base_angle_list_vehicles[instance_idx].append(random_interval([0, 360]))      # FIXME!
                # raise NotImplementedError

            out_ct += 1

            if out_ct > 10 or (
                    no_sample_idx is not None and idx in no_sample_idx):  # Not able to get the valid idx or no need to sample.
                if out_ct > 10:
                    print("Warning! We have droped an invalid idx: %d" % idx)
                else:
                    print("We skip invalid idx :%d" % idx)
                final_fail_flag = True
                break

            if fail_flag:
                continue

            else:
                print("Have tried %d times. idx: %d" % (out_ct, idx))
                break

        for instance_idx in range(instances_n):
            world_coord_list_vehicles[instance_idx].append(temp[instance_idx])
            base_angle_list_vehicles[instance_idx].append(random_interval([0, 360]))

        if not final_fail_flag:  # Mean the index is valid.
            valid_idx.append(idx)
        else:
            invalid_idx.append(idx)

    traces = []
    angles = []
    ego_delta = render_poses[-1]-render_poses[0]
    ego_delta = ego_delta[:3,3]
    ego_angle = math.atan2(ego_delta[1],ego_delta[0])
    # import pdb; pdb.set_trace()
    for instance_idx in range(instances_n):
        drive_angle = ego_angle+np.pi
        # drive_angle = base_angle_list_vehicles[instance_idx][0]/180*np.pi
        delta = np.array([np.cos(drive_angle), np.sin(drive_angle),0])
        ve = 10*0.1
        start_cd = world_coord_list_vehicles[instance_idx][0]
        trace = []
        angle = []
        for i in range(n_images):
            trace.append(start_cd+ve*i*delta)
            angle.append((drive_angle*180/np.pi+360)%360)
        traces.append(trace)
        angles.append(angle)

    world_coord_list_vehicles = traces
    base_angle_list_vehicles = angles
    # import pdb; pdb.set_trace()
    valid_idx = list(range(n_images))
    invalid_idx = []
    new_render_poses = render_poses[valid_idx, ...]
    new_world_coord_list_vehicles = [np.array(_)[valid_idx].tolist() for _ in world_coord_list_vehicles]
    new_base_angle_list_vehicles = [np.array(_)[valid_idx].tolist() for _ in base_angle_list_vehicles]
    new_n_images = len(valid_idx)

    return new_world_coord_list_vehicles, new_base_angle_list_vehicles, new_render_poses, new_n_images, bev_result, valid_idx, invalid_idx



#### Get masks from intrinsic and meshes. ####
def get_mask_from_mesh_use_inv(mesh,
                               resolution,
                               intrinsic):
    '''
    mesh should be in camera coordinate.
    resolution: (W, H)
    '''
    fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
    W, H = resolution
    ray_d = get_ray_directions(W, H, fx, fy, cx, cy).detach().cpu().numpy()
    ray_d = ray_d.reshape((-1,3))
    ray_o = np.zeros_like(ray_d)
    ray_o, ray_d = torch.tensor(ray_o).cuda(), torch.tensor(ray_d).cuda()
    
    # hit = mesh.ray.intersects_any(ray_o, ray_d)
    # mask = hit.reshape((H, W))    
    RT = raytracing.RayTracer(mesh.vertices, mesh.faces)
    intersections, face_normals, depth = RT.trace(ray_o, ray_d)
    depth = depth.detach().cpu().numpy()
    mask = ((depth < 100.).astype(np.uint8) * 255).reshape((H, W))    # No intersection means depth=max_depth=100.
    mask = np.uint8(mask)[..., None].repeat(3, axis=-1)
    return mask


def get_mask_from_mesh(mesh,
                       resolution,
                       intrinsic):
    '''
    mesh should be in camera coordinate.
    resolution: (W, H)
    '''
    def project(points_3D):
        points_2D = intrinsic @ points_3D
        points_2D = points_2D[:2, :] / points_2D[2, :]
        return points_2D
    
    def mask_outside(points_2D):
        pt_mask = np.ones_like(points_2D[0], dtype=bool)
        pt_mask = np.logical_and(points_2D[0] > 0, pt_mask)
        pt_mask = np.logical_and(points_2D[0] < W, pt_mask)
        pt_mask = np.logical_and(points_2D[1] > 0, pt_mask)
        pt_mask = np.logical_and(points_2D[1] < H, pt_mask)
        return points_2D[:, pt_mask]
    
    def points_to_mask(points_2D):
        mask_mat = np.zeros((H,W,3), dtype=np.uint8)
        x_array, y_array = points_2D[0, ...].astype(np.uint16), points_2D[1, ...].astype(np.uint16)
        mask_mat[y_array, x_array, ...] = 255
        return mask_mat
    
    def interpolate(mask_mat,
                    r=3,
                    iter=3):
        h, w, _ = mask_mat.shape
        mask_mat = np.pad(mask_mat, ((h, h), (w, w), (0, 0)))   # Padding for preventing corner cases on edges.
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r))
        mask_mat = cv2.morphologyEx(mask_mat, cv2.MORPH_CLOSE, close_kernel, iterations=iter)
        return mask_mat[h:2*h, w:2*w, ...]     # Please check the shape.    
            
            
    if intrinsic.shape == (4,4):
        intrinsic = intrinsic[:3,:3]
    W, H = resolution
    
    points_3D = mesh.vertices.T
    points_2D = project(points_3D)
    points_2D = mask_outside(points_2D)
    mask = points_to_mask(points_2D)
    mask = interpolate(mask)
    
    return mask


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True, resolution_level=1):
    # Abandoned.
    pixel_center = 0.5 if use_pixel_centers else 0
    l = resolution_level
    
    if resolution_level == 1:
        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32) + pixel_center,
            np.arange(H, dtype=np.float32) + pixel_center,
            indexing='xy'
        )
        i, j = torch.from_numpy(i), torch.from_numpy(j)
    
    else:
        tx = torch.linspace(0, W * l - 1, W)
        ty = torch.linspace(0, H * l - 1, H)      # Note that W and H have been scaled.
        i, j = torch.meshgrid(tx, ty, indexing='xy')    

    directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions    


