import math
from pickletools import uint8
import cv2
import argparse
import sys
sys.path.append('../')
import raytracing.raytracing as raytracing
# import raytracing
import numpy as np
import torch
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
from os import listdir
from os.path import join
from math import atan2, pi, sin, cos
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R 
from nuscenes.utils.geometry_utils import view_points
import trimesh
from copy import deepcopy

from ip_utils import get_bound, trans, scale, set_diff
from rasterizer import rasterize


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


def get_im(idx, bg_dir, valid_idx_list=None):
    idx = idx if valid_idx_list is None else valid_idx_list[idx]
    im_path = join(bg_dir, 'rgb', '%05d.png' % idx)
    return Image.open(im_path)
    return Image.fromarray(np_file[idx, ...])


def get_depth(idx, bg_dir, valid_idx_list=None):
    idx = idx if valid_idx_list is None else valid_idx_list[idx]
    depth_path = join(bg_dir, 'depth', '%05d.png' % idx)
    depth_mat = np.array(Image.open(depth_path))
    depth_mat = depth_mat / 256.
    return depth_mat
    return np_file[idx, ...]


def get_semantic(idx, bg_dir, valid_idx_list=None):
    idx = idx if valid_idx_list is None else valid_idx_list[idx]
    sem_path = join(bg_dir, 'semantic', '%05d.png' % idx)
    sem_mat = np.array(Image.open(sem_path))
    return sem_mat
    return np_file[idx, ...]


def get_instance_im(idx, sc, fg_dir, mode="SNeRF"):
    '''
    Get the vehicle image.
    '''
    if mode == "NeuS":
        im = Image.open(join(fg_dir, 'image', '%06d.png' % idx))
        mask_im = Image.open(join(fg_dir, 'mask', '%06d.png' % idx))
        
    else:
        im = Image.open(join(fg_dir, 'vehicles_img', '%d.png' % idx))
        mask_im = Image.open(join(fg_dir, 'vehicles_mask', '%d.png' % idx))
        
    # Scale.
    mask_im_mat = np.array(mask_im)
    im_mat = np.array(im)
    if sc != 1:
        x, y = im.size
        im_mat = cv2.resize(im_mat, (int(sc*x), int(sc*y)))
        mask_im_mat = cv2.resize(mask_im_mat, (int(sc*x), int(sc*y)))
    
    mask_im = Image.fromarray((mask_im_mat > 230).astype(np.uint8) * 255)   # Binary.
    im = Image.fromarray(im_mat)
    
    return im, mask_im


def get_scale(idx, fg_dir):
    scales = np.load(join(fg_dir, 'scale.npy'))
    return scales[0]


#### Geometry utils. ####
def project(points_3D,
            c2w,
            P):
    '''
    3D to 2D.
    Returns:
        An array with shape of [3, N].
    '''
    world_coords = np.concatenate((points_3D, np.ones((1, points_3D.shape[1]))), axis=0)
    cam_coords = np.linalg.inv(c2w) @ world_coords
    plane_coords = P @ cam_coords[:3, ...]
    return plane_coords


def find_position_rgb(bg_dir,
                      idx,
                      target_coord,
                      valid_idx_list=None):

    #### Get the sample. ####
    world_2_cam, P = get_calib(idx, bg_dir)
    im = get_im(idx, bg_dir, valid_idx_list)
    #########################
    
    #### Get the position. ####
    world_coord = np.array(target_coord).reshape((3, 1))
    world_coord = np.concatenate([world_coord, [[1]]], axis=0)
    # World -> Cam.
    cam_coord = world_2_cam @ world_coord
    # Projection.
    plane_coord = view_points(cam_coord[:3, :], P, normalize=True)[:2, :]

    ###########################
    
    return plane_coord, im


# ## ### ####  Mark. #### ### ## #

############# Abandoned ##############
def find_angle(bg_dir,
               idx,
               base_angle):
            
    #### Get the sample. ####
    world_2_cam, P = get_calib(idx, bg_dir)
    #########################    
    
    #### Get the angle. ####
    view_cam_coord = np.array([0, 0, 1]).reshape((3, 1))
    view_cam_coord = np.hstack((view_cam_coord, np.zeros((3, 1))))
    view_cam_coord = np.concatenate([view_cam_coord, [[1, 1]]], axis=0)
    # camear -> world
    cam_2_world = np.linalg.inv(world_2_cam)
    view_world_coord = cam_2_world @ view_cam_coord
    
    x, y = (view_world_coord[:, :1] - view_world_coord[:, 1:]).reshape((-1, ))[:2]
    angle = atan2(y, x) / (pi/2) * 90
    ########################
    
    angle = base_angle - angle
    angle = 360 + angle if angle < -180 else angle
    angle = angle - 360 if angle > 180 else angle
    
    return angle
######################################


############# Abandoned ##############
def find_camcalib(bg_dir,
                  idx,
                  target_coord,
                  angle):
    
    #### Get the sample. ####
    world_2_cam, P = get_calib(idx, bg_dir)
    # im = get_im(idx, channel)    
    #########################

    #### Get the rotation. ####
    rotation = R.from_euler('y', angle, degrees=True).as_matrix()
    ###########################

    #### Get the translation. ####
    world_coord = np.array(target_coord).reshape((3, 1))
    world_coord = np.concatenate([world_coord, [[1,]]], axis=0)
    # World -> Cam.
    cam_coord = world_2_cam @ world_coord
    translation = cam_coord[:3, :]
    ##############################
    
    #### Get the matrix. ####
    final_mat = np.eye(4)
    final_mat[:3, :3] = rotation
    final_mat[:3, 3:4] = translation
    #########################
    
    return np.linalg.inv(final_mat), P
########################################


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
    

def fuse(idx,
         bg_dir,
         bbox_center,
         theta,
         depth_mat,
         semantic_mat,
         background_im,
         vehicle_im,
         mask_im,
         mesh,
         category="vehicle"):
    '''
    Insert the vehicle (foreground) into background.
    Args:
        background_im, vehicle_im, mask_im: PIL.Image instance.
    '''    
    
    # Note that: This function also needs to handle the occulusion.
    # It need to be apllied in the future. #    # Already finish!! #
    # mesh = process_ply_to_camera_coord(idx, bg_dir, mesh, bbox_center, theta)
    mesh_world_coord = process_ply(mesh.copy(), bbox_center, theta)
    mesh_cam_coord = process_ply_to_camera_coord(idx, bg_dir, mesh.copy(), bbox_center, theta)
    
    bg_im_mat = np.array(background_im)
    vehicle_im_mat = np.array(vehicle_im)
    mask_im_mat = np.array(mask_im)
    
    #### Paste. ####
    # idx = np.where(mask_im_mat > 0)
    # bg_im_mat[idx] = vehicle_im_mat[idx]
    
    bg_im_mat, depth_mat, semantic_mat, mask_im_mat, occlution_per = handle_occlusion_paste(idx,
                                                                                            bg_dir,
                                                                                            bg_im_mat,
                                                                                            vehicle_im_mat,
                                                                                            mask_im_mat,
                                                                                            depth_mat,
                                                                                            semantic_mat,
                                                                                            mesh_world_coord,
                                                                                            mesh_cam_coord,
                                                                                            category)
    ################
    bbox_result = get_bbox_result(idx, bg_dir, mesh_cam_coord, bbox_center, theta, occlution_per, category)    
    
    #### Handle lighting. ####
    bg_im_mat = handle_lighting(bg_im_mat, mask_im_mat)
    ##########################
    
    return Image.fromarray(bg_im_mat), depth_mat, semantic_mat, bbox_result, occlution_per, Image.fromarray(mask_im_mat)


def get_bound_im(mask_im,
                 r):
    
    mask_im_mat = np.array(mask_im)
    
    # bound = get_bound(mask_im_mat[..., 0] > 0, r)
    # bound = bound.astype(np.uint8)
    # bound = bound * 255
    # bound = np.expand_dims(bound, axis=2).repeat(3, axis=2)    
    
    mask_im_mat = mask_im_mat[..., 0]
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1,r), max(1,r)))
    larger_mask = cv2.dilate(mask_im_mat, kernal, iterations=1)
    smaller_mask = cv2.erode(mask_im_mat, kernal, iterations=1)
    
    base_bound = np.logical_xor(larger_mask, smaller_mask)
    base_bound = base_bound[..., None].repeat(3, -1).astype(np.uint8) * 255
    
    return Image.fromarray(base_bound)


def fuse_bound_and_im(fuse_im,
                      bound_im):
    
    fuse_im_mat = np.array(fuse_im)
    bound_im_mat = np.array(bound_im)
    
    fuse_im_mat[bound_im_mat > 0] = 0
    
    return Image.fromarray(fuse_im_mat)


def fuse_bound(total_mask_im,
               total_bound_im,
               bound_im,
               mask_im):
    
    def set_diff(array_A, array_B):
        '''
        return array_A - array_B.
        '''
        return np.logical_and(array_A, np.logical_not(np.logical_and(array_A, array_B)))
    
    total_mask_im_mat = np.array(total_mask_im) > 0
    total_bound_im_mat = np.array(total_bound_im) > 0
    bound_im_mat = np.array(bound_im) > 0
    mask_im_mat = np.array(mask_im) > 0
    
    # 1. bound_im_mat - total_mask_im_mat.
    bound_im_mat = set_diff(bound_im_mat, total_mask_im_mat)
    # 2. total_bound_im_mat - mask_im_mat.
    total_bound_im_mat = set_diff(total_bound_im_mat, mask_im_mat)
        
    result_bound_im_mat = np.logical_or(bound_im_mat, total_bound_im_mat).astype(np.uint8) * 255

    return Image.fromarray(result_bound_im_mat)

def scale_intrinsic(P,
                    sc):
    P[0, 0] = P[0, 0] * sc
    P[1, 1] = P[1, 1] * sc
    P[0, 2] = P[0, 2] * sc
    P[1, 2] = P[1, 2] * sc
    return P


def decode_image(fg_dir,
                 idx,
                 P,
                 plane_coord,
                 render_factor,
                 inverse=False,
                 sc=None,
                 mode="SNeRF",
                 align=False,
                 category="vehicle",
                 meta_data={}):
    
    if inverse == True:
        idx_d = 299 - idx
    else:
        idx_d = idx

    # mesh render,  the same as person
    if meta_data.get('mesh_render', 0)>0:
        instance_im, mask_im = get_instance_im(idx_d,
                                               sc=1,  # ori 4//render_factor
                                               fg_dir=fg_dir,
                                               mode=mode)
        return instance_im, mask_im
        
    if category == "person":
        instance_im, mask_im = get_instance_im(idx_d,
                                               sc=4//render_factor,
                                               fg_dir=fg_dir,
                                               mode=mode)        
        return instance_im, mask_im 
        
    if mode == "SNeRF":
        instance_im, mask_im = get_instance_im(idx_d,
                                               sc=4//render_factor,
                                               fg_dir=fg_dir,
                                               mode=mode)
    elif mode == "NeuS":
        # if category != "vehicle":
        #     instance_im, mask_im = get_instance_im(idx_d,
        #                                         sc=1//render_factor,
        #                                         fg_dir=fg_dir,
        #                                         mode=mode)     
        # else:
        #     instance_im, mask_im = get_instance_im(idx_d,
        #                                         sc=4//render_factor,
        #                                         fg_dir=fg_dir,
        #                                         mode=mode)       

        instance_im, mask_im = get_instance_im(idx_d,
                                               sc=1//render_factor,
                                               fg_dir=fg_dir,
                                               mode=mode)        

    if not sc:
        sc = get_scale(idx, fg_dir=fg_dir)      # Help to scale.

    coord = P[:2, 2]
    instance_im, mask_im = scale(instance_im,
                                 mask_im,
                                 coord,
                                 scale=sc)
        
    instance_im, mask_im = trans(instance_im,
                                 mask_im,
                                 coord,
                                 plane_coord) 
    
    return instance_im, mask_im


#### About Mesh. ####
def read_obj(path, mesh_sc, mode="SNeRF"):
    with open(path) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                break       
    points = np.array(points)
    points = points.T

    if mode == "SNeRF":
        # Rotate. y -> z, x -> y, z -> x.
        Rot = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        points = Rot @ points * mesh_sc   # Scale.
    else:
        points = points * mesh_sc

    return points, np.zeros((3,1))


def read_ply(path, mesh_sc):
    
    mesh = trimesh.load(path)
    
    points = mesh.vertices.T
    points = points * mesh_sc
    
    return points, np.zeros((3,1))


def process_mesh(mesh_points,
                 center,
                 bbox_center,
                 theta,
                 bias):
    '''
    mesh_points: An array with shape of [3, N].
    Returns: An array with shape of [3, N] and center.
    '''
    rot = R.from_euler('z', theta, degrees=True).as_matrix()
    mesh_points = rot @ mesh_points + np.array(bbox_center).reshape((3,1))
    center = rot @ center + np.array(bbox_center).reshape((3,1))
    # Bias.
    mesh_points = mesh_points + rot @ np.array(bias).reshape((3,1))
    center = center + rot @ np.array(bias).reshape((3,1))
    return mesh_points, center


def process_ply(ply_mesh,
                bbox_center,
                theta):
    
    tran_matrix = np.eye(4).astype(np.float32)
    rot = R.from_euler('z', theta, degrees=True).as_matrix()
    trans = np.array(bbox_center)
    
    tran_matrix[:3, :3] = rot
    tran_matrix[:3, 3] = trans
    
    ply_mesh.apply_transform(tran_matrix)  
    return ply_mesh


def process_ply_to_camera_coord(idx,
                                bg_dir,
                                ply_mesh,
                                bbox_center,
                                theta):
    '''
    bbox_center, theta are in world global coord.
    The result mesh is under camera coord.
    '''
    w2c, P = get_calib(idx, bg_dir)
    ply_mesh = process_ply(ply_mesh, bbox_center, theta)
    ply_mesh.apply_transform(w2c)
    
    return ply_mesh


def get_vertix(mesh_points):
    x_max = mesh_points[:, np.argmax(mesh_points[0, :])]
    x_min = mesh_points[:, np.argmin(mesh_points[0, :])]
    y_max = mesh_points[:, np.argmax(mesh_points[1, :])]
    y_min = mesh_points[:, np.argmin(mesh_points[1, :])]
    z_max = mesh_points[:, np.argmax(mesh_points[2, :])]
    z_min = mesh_points[:, np.argmin(mesh_points[2, :])]
    points_3D = np.stack([x_max, x_min, y_max, y_min, z_max, z_min], axis=0)
    return points_3D.T


def get_bbox_result(idx,
                    bg_dir,
                    ply_mesh,
                    bbox_center,
                    theta,
                    occlution_per,
                    category):
    '''
    Return a dict:
        {
            'category', 'truncated', 'occlution',
            'alpha', 'xmin', 'ymin', 'xmax', 'ymax',
            'pos_x', 'pos_y', 'pos_z', 'rot_y'
        }
    Note:
        We save it as kitti format, that means everything in camera coord.
    
    '''
    def occlution_level():
        if occlution_per < .01:
            return 0
        if occlution_per < .5:
            return 1
        if occlution_per < .99:
            return 2
        return 3    
    
    # bbox_result = {}
    # bbox_result['rot_z'] = theta
    # bbox_result['pos_x'], bbox_result['pos_y'], bbox_result['pos_z'] = bbox_center
    
    # _, extents = trimesh.bounds.oriented_bounds(ply_mesh)
    # bbox_result['height'], bbox_result['width'], bbox_result['length'] = extents
    
    # return bbox_result
    
    category_cvt = {
        "vehicle": "Car", "person": "Pedestrain", "object": "Object", "bicycle": "Bicycle", "motorcycle": "Motorcycle"
    }
    
    bbox_result = {}
    w2c, P = get_calib(idx, bg_dir)
    
    # Calculate pos.
    bbox_center = np.concatenate([np.array(bbox_center).reshape((3,1)), [[1]]], axis=0)
    bbox_center_cam_coord = (w2c @ bbox_center)[:3, 0]
    
    # Calculate rot_y.
    # Note the rotation is: rot_y @ rot_axis = rot_w2c @ rot_z
    rot_z = R.from_euler('z', theta, degrees=True).as_matrix()
    rot_w2c = w2c[:3, :3]
    rot_axis = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    rot_y = rot_w2c @ rot_z @ rot_axis.T
    y_angle = R.from_matrix(rot_y).as_euler('yzx', degrees=False)[0]    
    
    bbox_result['category'] = category_cvt[category]
    bbox_result['occlution'] = occlution_level()
    _, extents = trimesh.bounds.oriented_bounds(ply_mesh)
    if category == "vehicle":
        bbox_result['height'], bbox_result['width'], bbox_result['length'] = extents    
    elif category == "bicycle":
        bbox_result['width'], bbox_result['height'], bbox_result['length'] = extents
    elif category == "person":
        bbox_result['length'], bbox_result['width'], bbox_result['height'] = extents
    elif category == 'motorcycle':
        bbox_result['width'], bbox_result['height'], bbox_result['length'] = extents
    bbox_result['pos_x'], bbox_result['pos_y'], bbox_result['pos_z'] = bbox_center_cam_coord
    # We set other annotation as 0.
    ####################################### Can append new information here. #######################################
    bbox_result['alpha'], bbox_result['xmin'], bbox_result['ymin'], bbox_result['xmax'], bbox_result['ymax'], bbox_result['truncated'] =\
        .0, .0, .0, .0, .0, .0
    ####################################### Can append new information here. #######################################

    ### ori
    # bbox_result["rot_y"] = y_angle
    ### todo: some thing wrong here
    bbox_result["rot_y"] = y_angle-math.pi/2
    bbox_result["confidence"] = 1

    return bbox_result


def save_bbox_result_for_one_frame(out_path,
                                   bbox_results_list):
    
    with open(out_path, 'w') as f:
        for res in bbox_results_list:
            rec = "%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f" % \
                (res["category"], res["truncated"], res["occlution"], res["alpha"],\
                    res["xmin"], res["ymin"], res["xmax"], res["ymax"],\
                    res["height"], res["width"], res["length"],\
                    res["pos_x"], res["pos_y"], res["pos_z"], res["rot_y"], res["confidence"])
            f.write(rec)
            f.write("\n")

#####################

#### Calculate the scale. ####
def cal_sc(idx,
           mesh_dir,
           fg_dir,
           bg_dir,
           target_coord,
           base_angle,
           inverse=False,
           mode="SNeRF"):
    '''
    Calculate the scale based on the real mesh.
    Returns:
        A scalar scale.
    '''
    # Read the data.
    if inverse:
        idx_d = 299 - idx
    else:
        idx_d = idx
    
    if mode == "SNeRF":
        mesh_points, center = read_obj(mesh_dir, mesh_sc=1)
    else:
        mesh_points, center = read_ply(mesh_dir, mesh_sc=1)
    _, mask_im = get_instance_im(idx_d, sc=4, fg_dir=fg_dir, mode=mode)
    mask_mat = np.array(mask_im)
    world_2_cam, P = get_calib(idx, bg_dir)
    c2w = np.linalg.inv(world_2_cam)
    
    # Process the mesh.
    mesh_points, center = process_mesh(mesh_points, center, target_coord, base_angle, bias=[0,0,0])
    points_2D = project(mesh_points, c2w, P)
    points_2D = points_2D[:2, ...] / points_2D[2, ...]
    
    # Calculate the scale.
    mesh_length = points_2D[0, ...].max() - points_2D[0, ...].min()
    i_list, j_list, _ = np.where(mask_mat > 0)
    if j_list.size == 0:
        return 1         # 別にいいですけど
    mask_length = j_list.max() - j_list.min()
    
    sc = mask_length / mesh_length
    
    return sc


#### Judge the occlusion. ####
def occlution_order(idx,
                    fg_data_dict,
                    vehicle_idx_list,
                    bg_dir,
                    mesh_list,
                    render_factor,
                    mode='NeuS',
                    align=True,
                    meta_data={}):
    '''
    Only support NeuS mode.
    '''
    
    def get_vehicle_mask_im(idx, vehicle_idx):
        w2c, P = get_calib(idx, bg_dir)
        world_coord = world_coord_vehicle_list[vehicle_idx]
        base_angle = base_angle_vehicle_list[vehicle_idx]
        mesh_dir = mesh_dir_vehicle_list[vehicle_idx]
        fg_dir = fg_dir_vehicle_list[vehicle_idx]
        category = category_list[vehicle_idx]
        
        plane_coord, im = find_position_rgb(bg_dir,
                                            idx,
                                            world_coord)    
        
        if mode == "SNeRF":
            inverse_flag = False
            sc = cal_sc(idx, mesh_dir, fg_dir, bg_dir, world_coord, base_angle, inverse_flag)          
            # If not want to use calculated sc, let sc = None.
            vehicle, mask_im = decode_image(fg_dir, idx, P, plane_coord, render_factor, inverse_flag, sc, mode=mode, category=category,
                                            meta_data=meta_data)
        else:
            inverse_flag = False
            sc = None
            if align:
                sc = cal_sc(idx, mesh_dir, fg_dir, bg_dir, world_coord, base_angle, inverse_flag, mode=mode)
            vehicle, mask_im = decode_image(fg_dir, idx, P, plane_coord, render_factor, inverse_flag, sc, mode=mode, align=align, category=category,
                                            meta_data=meta_data)
            
        return mask_im        
    
    def intersect_position_z(ray_o, ray_d, mesh):
        tri = mesh.triangles_center
        tri_index = mesh.ray.intersects_first(ray_o, ray_d)
        position_z = tri[tri_index, 2]
        return position_z
    
    def occlution_for_two_vehicle(mesh_A, mesh_B, mask_A, mask_B):
        '''
        Judge whether A occlude B or not.
        '''
        w2c, P = get_calib(idx, bg_dir)
        
        # First find whether occlution.
        mask_A, mask_B = mask_A > 0, mask_B > 0
        intersection_mask = np.logical_and(mask_A, mask_B)
        if np.sum(intersection_mask) == 0:   # No occlution.
            return None    # No way to compare.
        
        # Use one ray to find out who cooludes.
        i_list, j_list, _ = np.where(intersection_mask > 0)
        c2w = np.linalg.inv(w2c)
        ray_point_2D = np.array([j_list.mean(), i_list.mean(), 1]).reshape((3,1))   # We set the depth as 1.
        ray_point_3D = np.linalg.inv(P) @ ray_point_2D
        ray_point_3D = np.concatenate((ray_point_3D, [[1,]]), axis=0)
        ray_point_3D = c2w @ ray_point_3D
        ray_point_3D = ray_point_3D[:3, 0]
        ray_o = c2w[:3, 3]
        ray_d = (ray_point_3D - ray_o) / np.linalg.norm(ray_point_3D - ray_o)
        ray_o, ray_d = ray_o.reshape((1,3)), ray_d.reshape((1,3))
        
        # Use trimesh to judge. 
        z_A, z_B = intersect_position_z(ray_o, ray_d, mesh_A)[0], intersect_position_z(ray_o, ray_d, mesh_B)[0]
        
        return z_A < z_B
                
    vehicles_n = len(vehicle_idx_list)
    
    mesh_dir_vehicle_list = [fg_data_dict[vehicle_idx]['mesh_dir'] for vehicle_idx in vehicle_idx_list]
    base_angle_vehicle_list = [fg_data_dict[vehicle_idx]['base_angle_list'][idx] for vehicle_idx in vehicle_idx_list]
    fg_dir_vehicle_list = [fg_data_dict[vehicle_idx]['fg_dir'] for vehicle_idx in vehicle_idx_list]
    world_coord_vehicle_list = [np.array(fg_data_dict[vehicle_idx]['world_coord_list'][idx]) for vehicle_idx in vehicle_idx_list]
    theta_list = [fg_data_dict[vehicle_idx]['base_angle_list'][idx] for vehicle_idx in vehicle_idx_list]
    category_list = [fg_data_dict[vehicle_idx]['category'] for vehicle_idx in vehicle_idx_list]
    
    # 1. Process meshes.
    mesh_list = [process_ply_to_camera_coord(idx, bg_dir, mesh_list[vehicle_idx], world_coord_vehicle_list[vehicle_idx], theta_list[vehicle_idx]) for vehicle_idx in vehicle_idx_list]
    mask_list = [np.array(get_vehicle_mask_im(idx, vehicle_idx)) for vehicle_idx in vehicle_idx_list]
    
    # 2. Build a DAG.
    A = np.zeros((vehicles_n, vehicles_n)).astype(np.uint8)
    for i in range(vehicles_n):
        for j in range(i+1, vehicles_n):
            mesh_A, mesh_B = mesh_list[i], mesh_list[j]
            mask_A, mask_B = mask_list[i], mask_list[j]
            A_occlude_B = occlution_for_two_vehicle(mesh_A, mesh_B, mask_A, mask_B)
            if A_occlude_B == True:    # B should appear first.
                A[j, i] = 1
            elif A_occlude_B == False: # A should appear first.
                A[i, j] = 1
            else:
                continue
    
    # 3. Topological sort.
    result_list = []
    while len(result_list) < vehicles_n:
        for i in range(vehicles_n):
            if i not in result_list and np.sum(A[:, i]) == 0:
                result_list.append(i)   # Output idx.
                for j in range(vehicles_n):   # Remove idx.
                    if A[i, j] == 1:
                        A[i, j] = 0
                break 
        else:
            print("We find a circle...")    
            raise RuntimeError("Occlusion judgment fails....")
            
    return result_list
    
    ##### Abandoned part. #####
    
    world_coord_vehicle_list = [np.array(fg_data_dict[vehicle_idx]['world_coord_list'][idx]).reshape((3,1)) for vehicle_idx in vehicle_idx_list]
    world_coord_vehicle_list = [np.concatenate((world_coord, [[1.,]]), axis=0) for world_coord in world_coord_vehicle_list]
    
    # Get the camera calibs.
    w2c, _ = get_calib(idx, bg_dir)
    
    # Now we can find the occlusion.
    camera_coord_vehicle_list = [w2c @ coord for coord in world_coord_vehicle_list]
    z_vehicle_list = [coord[2,0] for coord in camera_coord_vehicle_list]
    sorted_id = sorted(range(len(z_vehicle_list)), key=lambda k: z_vehicle_list[k], reverse=True)    # Deeper one appears earlier.
    
    return sorted_id


def handle_occlusion_paste(idx,
                           bg_dir,
                           bg_im_mat,
                           vehicle_im_mat,
                           mask_im_mat,
                           depth_mat,
                           semantic_mat,
                           mesh_in_world_coord,
                           mesh_in_cam_coord,
                           category="vehicle"):

    '''Paste but consider the occulution between fg and bg.'''
    
    def intersect_position_z(ray_o, ray_d, max_size=1000):     # FIXME
        
        ########################################################################
        # tri = mesh_in_cam_coord.triangles_center             # Using Trimesh.
        # n_steps = ((ray_o.shape[0] - 1) // max_size) + 1
        # position_z = []
        # for step in range(n_steps):
        #     ray_o_batch, ray_d_batch = ray_o[step*max_size : (step+1)*max_size, :], ray_d[step*max_size : (step+1)*max_size, :]
        #     tri_index_batch = mesh_in_cam_coord.ray.intersects_first(ray_o_batch, ray_d_batch)
        #     position_z_batch = tri[tri_index_batch, 2]
        #     position_z.append(position_z_batch)
        # position_z = np.concatenate(position_z, axis=0)
        # return position_z    
        ########################################################################
        
        ray_o, ray_d = torch.tensor(ray_o).cuda(), torch.tensor(ray_d).cuda()
        
        _mesh = mesh_in_cam_coord.copy()
        _mesh.apply_transform(np.array([                
            [ 1, 0, 0, 0],
            [ 0, -1, 0, 0],
            [ 0, 0, -1, 0],
            [ 0, 0, 0, 1]
        ]))
        
        RT = raytracing.RayTracer(_mesh.vertices, _mesh.faces)
        intersections, face_normals, depth = RT.trace(ray_o, ray_d)
        depth = depth.detach().cpu().numpy()
        depth[depth >= 99.] = .1               # Max depth = 100. Depth > max_depth means no intersection.
        return depth
        
    
    # def get_depth_from_mesh(i, j):
        
    #     w2c, P = get_calib(idx, bg_dir)
    #     c2w = np.linalg.inv(w2c)
    #     ray_point_2D = np.array([j, i, 1]).reshape((3,1))   # We set the depth as 1.
    #     ray_point_3D = np.linalg.inv(P) @ ray_point_2D
    #     ray_point_3D = np.concatenate((ray_point_3D, [[1,]]), axis=0)
    #     ray_point_3D = c2w @ ray_point_3D
    #     ray_point_3D = ray_point_3D[:3, 0]
    #     ray_o = c2w[:3, 3]
    #     ray_d = (ray_point_3D - ray_o) / np.linalg.norm(ray_point_3D - ray_o)
    #     ray_o, ray_d = ray_o.reshape((1,3)), ray_d.reshape((1,3))        

    #     return intersect_position_z(ray_o, ray_d)
    
    
    def get_depth_from_mesh_batch(i_list, j_list):
        
        ###########################################  # Old version. Have some problem.
        # w2c, P = get_calib(idx, bg_dir)
        # c2w = np.linalg.inv(w2c)
        # ray_points_2D = np.array([j_list, i_list, np.ones_like(j_list)])
        # ray_points_3D = np.linalg.inv(P) @ ray_points_2D
        # ray_points_3D = np.concatenate((ray_points_3D, np.ones_like(ray_points_3D[:1, ...])), axis=0)
        # ray_points_3D = c2w @ ray_points_3D                           # World coordinate.
        # ray_o = c2w[:3, 3, None].repeat(j_list.size, axis=1)          # World coordinate.
        # ray_d = (ray_points_3D - ray_o) / np.linalg.norm(ray_points_3D - ray_o, axis=0)
        # ray_o, ray_d = ray_o.reshape((-1,3)), ray_d.reshape((-1,3))
        ###########################################
        
        w2c, P = get_calib(idx, bg_dir)
        fx, fy, cx, cy = P[0,0], P[1,1], P[0,2], P[1,2]
        ray_d = get_ray_d_from_xy(j_list, i_list, fx, fy, cx, cy)
        ray_o = torch.zeros_like(ray_d)
        
        return intersect_position_z(ray_o, ray_d)
    
    i_list, j_list = np.where(mask_im_mat[..., 0] > 0)
    pixel_n = i_list.size
    
    valid_pixel_count = 0
    
    ########################################################################
    # for pixel_idx in range(pixel_n):                         # Pixel-wise. Low efficiency.
    #     i, j = i_list[pixel_idx], j_list[pixel_idx]
    #     bg_depth = depth_mat[i, j]
    #     fg_depth = get_depth_from_mesh(i, j)
    #     if fg_depth < bg_depth or semantic_mat[i, j] == 0\
    #         or semantic_mat[i, j] == 1\
    #         or semantic_mat[i, j] == 8:    # That means foreground should "overwrite" background. semantic == 0 is the road.
                
    #         bg_im_mat[i, j, :] = vehicle_im_mat[i, j, :]
    #         depth_mat[i, j] = fg_depth
    #         semantic_mat[i, j] = 13 if category == "vehicle" else 11
    #         valid_pixel_count += 1
            
    #     else:
    #         mask_im_mat[i, j, ...] = 0   
    ########################################################################s
    
    semantic_dict = {
        "vehicle": 13, "person": 11, "object": 0, "bicycle": 18, "motorcycle": 17
    }
    
    bg_depth = depth_mat[i_list, j_list]

    if category == "person":                                     # Some wrong for persons...
        fg_depth = -1 * np.ones_like(bg_depth)

    else:
        
        fg_depth = get_depth_from_mesh_batch(i_list, j_list)     # Use trimesh / raytracing.
        
        '''
        # To debug, use:
        if True:
            temp_vis = np.zeros(mask_im_mat.shape[:2])
            temp_vis[i_list, j_list] = fg_depth
            temp_vis = (temp_vis * 256).astype(np.uint16)
            
            w2c, P = get_calib(idx, bg_dir)
            H, W = mask_im_mat.shape[:2]
            fx, fy, cx, cy = P[0,0], P[1,1], P[0,2], P[1,2]
            ray_d = get_ray_directions(W, H, fx, fy, cx, cy)
            ray_o = np.zeros_like(ray_d)
            depth = intersect_position_z(ray_o, ray_d)
            
            mask = (depth < 100.).reshape((H,W))
            mask = (mask.astype(np.uint8) * 255)[..., None].repeat(3, -1)
            Image.fromarray(mask).save("./temp_vis.png")
            import pdb; pdb.set_trace()
        '''
        
        
        ########################################################################
        # resolution = (mask_im_mat.shape[1], mask_im_mat.shape[0])      # Use rasterizing.
        # w2c, P = get_calib(idx, bg_dir)
        # _, depth = rasterize(mesh_in_world_coord, resolution, P, w2c)
        
        # # Refine.
        # valid_depth = (depth > 0)
        # to_refine_i_list, to_refine_j_list = np.where((~valid_depth & mask_im_mat[..., 0].astype(bool)) == True)
        # if to_refine_i_list.size != 0:     
        #     refine_depth = get_depth_from_mesh_batch(to_refine_i_list, to_refine_j_list)
        #     depth[to_refine_i_list, to_refine_j_list] = refine_depth
        
        # fg_depth = depth[i_list, j_list]
        ########################################################################

    # except:
    # fg_depth = -1 * np.ones_like(bg_depth)             # Error occurs! 
    
    semantic = semantic_mat[i_list, j_list]
    valid_mask = np.zeros_like(bg_depth, dtype=bool)
    
    valid_mask = np.logical_or(valid_mask, fg_depth < bg_depth)
    valid_mask = np.logical_or(valid_mask, semantic == 0)
    valid_mask = np.logical_or(valid_mask, semantic == 1)
    valid_mask = np.logical_or(valid_mask, semantic == 8)
    invalid_mask = np.logical_not(valid_mask)
    
    valid_i_list, valid_j_list = i_list[valid_mask], j_list[valid_mask]
    invalid_i_list, invalid_j_list = i_list[invalid_mask], j_list[invalid_mask]
    
    bg_im_mat[valid_i_list, valid_j_list, ...] = vehicle_im_mat[valid_i_list, valid_j_list, ...]
    depth_mat[valid_i_list, valid_j_list] = fg_depth[valid_mask]
    # semantic_mat[valid_i_list, valid_j_list] = 13 if category == "vehicle" else 11
    semantic_mat[valid_i_list, valid_j_list] = semantic_dict[category]
    mask_im_mat[invalid_i_list, invalid_j_list] = 0
            
    valid_pixel_count = valid_mask.sum()
    occlution_per = 1 - valid_pixel_count / (pixel_n + 1)
    
    return bg_im_mat, depth_mat, semantic_mat, mask_im_mat, occlution_per


#### Handle the lighting. ####
def handle_lighting(im_mat,
                    mask_mat):
    
    # Read the data.
    im_mat = cv2.cvtColor(im_mat, cv2.COLOR_RGB2HSV)
    H, W, _ = im_mat.shape
    
    # Get the local region.
    i_fg_list, j_fg_list = np.where(mask_mat[..., 0] > 0)
    i_bg_list, j_bg_list = np.where(mask_mat[..., 0] == 0)
    if i_fg_list.size == 0:
        return cv2.cvtColor(im_mat, cv2.COLOR_HSV2RGB)
    
    i_min, i_max, j_min, j_max = i_fg_list.min(), i_fg_list.max(), j_fg_list.min(), j_fg_list.max()
    i_length, j_length = i_max - i_min, j_max - j_min
    i_min, i_max = max(0, i_min - i_length//2 - 1), min(H-1, i_max + i_length//2 + 1)
    j_min, j_max = max(0, j_min - j_length//2 - 1), min(W-1, j_max + j_length//2 + 1)
    
    mask = np.ones_like(i_bg_list, dtype=bool)
    mask = np.logical_and(mask, i_bg_list >= i_min)
    mask = np.logical_and(mask, i_bg_list <= i_max)
    mask = np.logical_and(mask, j_bg_list >= j_min)
    mask = np.logical_and(mask, j_bg_list <= j_max)
    i_bg_list, j_bg_list = i_bg_list[mask], j_bg_list[mask]

    # Compute the individual part.
    bg_v_data = im_mat[i_bg_list, j_bg_list, 2]
    fg_v_data = im_mat[i_fg_list, j_fg_list, 2]
    bg_mean, bg_var = bg_v_data.mean(), bg_v_data.var()
    fg_mean, fg_var = fg_v_data.mean(), fg_v_data.var()
    
    # Transform.
    if not fg_var < 1e-7:    # No transform.
        # fg_v_data = ((fg_v_data - fg_mean) / np.sqrt(fg_var) * np.sqrt(bg_var) + bg_mean)
        fg_v_data = fg_v_data - fg_mean + bg_mean
        fg_v_data = np.clip(fg_v_data, 0, 255)
        fg_v_data = fg_v_data.astype(np.uint8)
    
    # Apply.
    im_mat[i_fg_list, j_fg_list, 2] = fg_v_data
    
    im_mat = cv2.cvtColor(im_mat, cv2.COLOR_HSV2RGB)
    
    return im_mat


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True, resolution_level=1):
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


def get_ray_d_from_xy(x, y, fx, fy, cx, cy, use_pixel_centers=True):
    
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    directions = torch.stack([(x - cx) / fx, -(y - cy) / fy, -torch.ones_like(x)], -1) # (H, W, 3)
    
    return directions