import cv2
import numpy as np
import imageio
import trimesh
from PIL import Image
from numpy import clip
from os.path import join
from math import sin, cos, pi
from scipy.spatial.transform import Rotation as R

#### Data loading utils. ####

def read_sc(dir, idx):
    file = np.load(join(dir, 'scale.npy'))
    return file[idx]

def read_calib(dir, idx):
    file = np.load(join(dir, 'target_poses.npy'))
    return file[idx, ...]

def read_P(dir, idx):
    file = np.load(join(dir, 'intrinsic.npy'))
    if len(file.shape) == 2:
        return file
    return file[idx, ...]

def read_bias(dir, idx):
    file = np.load(join(dir, 'calibs.npy'))
    return file[idx, 2, 3]

def read_im(dir, idx):
    return Image.open(join(dir, 'image', '%05d.png' % idx))
    # return Image.open(join(dir, 'image', '%07d_fuse_im.png' % idx))

def read_mask_mat(dir, idx, vehicle_idx, erode=1):
    '''
    Return a mask_mat, with shape of [H, W, 3].
    '''
    # mask_mat = np.array(Image.open(join(dir, 'mask_%d' % vehicle_idx, '%07d_mask.png' % idx)))
    mask_mat = np.array(Image.open(join(dir, 'mask_%d' % vehicle_idx, '%05d.png' % idx)))
    mask_mat = cv2.erode(mask_mat, cv2.getStructuringElement(cv2.MORPH_CROSS, (erode, erode)), iterations=2)
    return mask_mat

def read_total_mask_mat(dir, idx, erode=1):
    '''
    Return a mask_mat (containing all of the vehicles), with shape of [H, W, 3].
    '''
    mask_mat = np.array(Image.open(join(dir, 'mask', '%05d.png' % idx)))
    mask_mat = cv2.erode(mask_mat, cv2.getStructuringElement(cv2.MORPH_CROSS, (erode, erode)), iterations=2)
    return mask_mat    

def read_depth_mat(dir, idx, vehicle_idx):
    '''
    Return a depth_mat, with shape of [H, W, 3].
    '''
    # depth_mat = np.array(Image.open(join(dir, 'depth_%d' % vehicle_idx, '%07d_depth_im.png' % idx)))
    depth_mat = np.array(Image.open(join(dir, 'depth_%d' % vehicle_idx, '%05d.png' % idx)))
    depth_mat = depth_mat.astype(np.float32) / 255 * 10      # Denormalize.
    # depth_mat = depth_mat * sc * base_sc                             # Scale the depth value.
    # bias = read_bias(dir, idx)
    # depth_mat = depth_mat * base_sc - bias
    return depth_mat

def read_mask_depth_mat(dir, idx, vehicle_idx, erode):
    '''
    Return a mask_mat and depth_mat.
    '''
    mask_mat = read_mask_mat(dir, idx, vehicle_idx, erode)
    depth_mat = read_depth_mat(dir, idx, vehicle_idx)

    depth_mat[np.where(mask_mat == 0)] = 0
    depth_mat = depth_mat - depth_mat.mean()
    return mask_mat, depth_mat


#### Geometry utils. ####

def get_all_2D_points(mask_mat,
                      depth_mat):
    '''
    Get all 2D points. 
    Returns:
        An array with shape of [3, N].
    '''
    y_array, x_array = np.where(mask_mat[..., 0] > 0)
    n_points = x_array.shape[0]
    
    depth_array = depth_mat[y_array, x_array, 0]                   # Only need to choose one channel.
    x_array = x_array.astype(np.float32) * depth_array
    y_array = y_array.astype(np.float32) * depth_array

    points = np.stack([x_array, y_array, depth_array], axis=0)     # Please check the shape and the values.
    return points


def inv_project(points_2D,
                c2w,
                P):
    '''
    2D to 3D, and convert to world's coordinates.
    Returns:
        An array with shape of [3, N].
    '''
    cam_coords = np.linalg.inv(P) @ points_2D
    cam_coords = np.concatenate((cam_coords, np.ones((1, cam_coords.shape[1]))), axis=0)
    world_coords = c2w @ cam_coords
    return world_coords[:3, ...]


def world_to_bbox(points_3D,
                  bbox_center):
    '''
    World coords to bbox coords.
    Returns:
        An array with shape of [3, N].
    '''
    return points_3D - np.array(bbox_center).reshape((3, 1))


def bbox_to_world(points_3D,
                  bbox_center):
    '''
    Bbox coords to world coords.
    Returns:
        An array with shape of [3, N].
    '''    
    return points_3D + np.array(bbox_center).reshape((3, 1))


def project_to_ground(points_3D,
                      pitch_angle,
                      yaw_angle,
                      ground_height):
    '''
    Project the 3D points to the ground plane.
    ground_height: The z-coordinate (in world's coordinates) of ground.
        *If ground_height is None, then we use the lowest 3D point's z-coordinate as ground_height.
    Returns:
        An array with shape of [3, N].
    '''
    if points_3D.size == 0:
        return points_3D
    if not ground_height:
        z_axis = points_3D[2, ...].copy()
        z_axis = np.sort(z_axis)
        ground_height = np.median(z_axis[:1])
        # print(ground_height)

    pitch_angle, yaw_angle = pitch_angle/180*pi, yaw_angle/180*pi

    light_vec = np.array([sin(pitch_angle)*cos(yaw_angle), sin(pitch_angle)*sin(yaw_angle), cos(pitch_angle)])
    coef = (points_3D[2, ...] - ground_height) / light_vec[2]
    points_3D = points_3D - light_vec.reshape((3,1)) * coef
    return points_3D                # Please check the z-coordinates.


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


def points_to_mask(points_2D,
                   mask_mat):
    '''
    Convert 2D points to mask_mat.
    Returns:
        An array with shape of [H, W, 3].
    '''
    def clip_points(array1, array2, min1, max1, min2, max2):
        mask = np.ones_like(array1, dtype=bool)
        mask = np.logical_and(mask, array1 > min1)
        mask = np.logical_and(mask, array1 < max1)
        mask = np.logical_and(mask, array2 > min2)
        mask = np.logical_and(mask, array2 < max2)        
        return array1[mask], array2[mask]

    h, w, _ = mask_mat.shape
    points_2D = points_2D / points_2D[2, :]                      # Normalize the 2D points. Please check the z-coordinates.
    shadow_mask_mat = np.zeros(mask_mat.shape, dtype=bool)
    x_array, y_array = points_2D[0, ...].astype(np.uint16), points_2D[1, ...].astype(np.uint16)
    x_array, y_array = clip_points(x_array, y_array, 0, w, 0, h)
    shadow_mask_mat[y_array, x_array, ...] = True
    return shadow_mask_mat.astype(np.uint8) * 255                # Please check the shape and the type.


def interpolate(shadow_mask_mat,
                r=3,
                iter=3):
    '''
    Interpolate the shadow mask with morphology methods.
    Returns:
        An array with shape of [H, W, 3].
    '''
    h, w, _ = shadow_mask_mat.shape
    shadow_mask_mat = np.pad(shadow_mask_mat, ((h, h), (w, w), (0, 0)))   # Padding for preventing corner cases on edges.
    
    # open_kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r))
    # shadow_mask_mat = cv2.morphologyEx(shadow_mask_mat, cv2.MORPH_OPEN, open_kernal, iterations=1)
    shadow_mask_mat = cv2.morphologyEx(shadow_mask_mat, cv2.MORPH_CLOSE, close_kernel, iterations=iter)

    return shadow_mask_mat[h:2*h, w:2*w, ...]     # Please check the shape.


def check_the_occlusion(mask_mat,
                        shadow_mask_mat):
    '''
    Seperate the shadow mask from the vehicle mask.
    Returns:
        An array with shape of [H, W, 3].
    '''
    # Binary.
    shadow_mask_mat, mask_mat = shadow_mask_mat > 0, mask_mat > 0
    # Setdiff operates.
    shadow_mask_mat = np.logical_and(shadow_mask_mat, np.logical_not(np.logical_and(shadow_mask_mat, mask_mat)))
    return shadow_mask_mat.astype(np.uint8) * 255


def shadow_refine(mask_mat,
                  shadow_mask_mat,
                  r=30,
                  iter=2):
    '''
    Fill the space between shadow and vehicle.
    Returns:
        An array with shape of [H, W, 3].
    '''
    # Binary.
    shadow_mask_mat, mask_mat = shadow_mask_mat > 0, mask_mat > 0
    # Fill the space.
    shadow_mask_mat = np.logical_or(shadow_mask_mat, mask_mat).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r))
    shadow_mask_mat = cv2.morphologyEx(shadow_mask_mat, cv2.MORPH_CLOSE, kernel, iterations=iter)
    # Setdiff operates.
    shadow_mask_mat = np.logical_and(shadow_mask_mat, np.logical_not(np.logical_and(shadow_mask_mat, mask_mat)))
    return shadow_mask_mat.astype(np.uint8) * 255


def shadow_fuse(im,
                shadow_mask_mat,
                scale,
                blur=None):
    '''
    Fuse the shadow and the image.
    Returns:
        A PIL.Image instance.
    '''
    im_mat = np.array(im)
    temp_layer = im_mat.copy()
    temp_layer[np.where(shadow_mask_mat > 0)] = 0

    # Blur the shadow.
    if blur:
        # blur_shadow = cv2.blur(shadow_mask_mat, ksize=(3,3))
        # temp_layer[np.where(shadow_mask_mat > 0)] = 255-blur_shadow[np.where(shadow_mask_mat > 0)]
        temp_layer = cv2.blur(temp_layer, ksize=(blur,blur))
        temp_layer[np.where(shadow_mask_mat == 0)] = im_mat[np.where(shadow_mask_mat == 0)]

    im_mat = cv2.addWeighted(temp_layer, scale, im_mat, 1-scale, gamma=0)
    return Image.fromarray(im_mat)


def median_filter(rgb_npy,
                  shadow_mask_npy,
                  win_size=(5,3,3)):
    '''
    Filter the shadow.
    win_size: (r_n, r_y, r_x)
    Returns:
        An array with shape of [N, H, W, 3].
    '''
    print("Median filter.")
    r_n, r_y, r_x = win_size
    N, H, W, _ = rgb_npy.shape
    result_npy = rgb_npy.copy()
    for c in range(3):
        bd_idx = np.where(shadow_mask_npy[..., c] > 0)
        n_idx, i_idx, j_idx = bd_idx
        n_pixels = n_idx.size
        print(n_pixels)
        
        for idx in range(n_pixels):
            n, i, j = n_idx[idx], i_idx[idx], j_idx[idx]
            neighbors = rgb_npy[clip(n-r_n//2,0,N):clip(n+r_n//2+1,0,N), clip(i-r_y//2,0,H):clip(i+r_y//2+1,0,H), clip(j-r_x//2,0,W):clip(j+r_x//2+1,0,W), c]
            result_npy[n, i, j, c] = np.median(neighbors)
            if idx % 10000 == 0:
                print(idx)  

    return result_npy  



#### Mesh utils. ####
def read_obj(path, mesh_sc):
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

    # Rotate. y -> z, x -> y, z -> x.
    Rot = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    points = Rot @ points * mesh_sc   # Scale.

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


def get_vertix(mesh_points):
    x_max = mesh_points[:, np.argmax(mesh_points[0, :])]
    x_min = mesh_points[:, np.argmin(mesh_points[0, :])]
    y_max = mesh_points[:, np.argmax(mesh_points[1, :])]
    y_min = mesh_points[:, np.argmin(mesh_points[1, :])]
    z_max = mesh_points[:, np.argmax(mesh_points[2, :])]
    z_min = mesh_points[:, np.argmin(mesh_points[2, :])]
    points_3D = np.stack([x_max, x_min, y_max, y_min, z_max, z_min], axis=0)
    return points_3D.T


def vis_bbox_center(im,
                    bbox_center,
                    c2w,
                    P):
    im_draw = ImageDraw.Draw(im)
    bbox_center = np.array(bbox_center).reshape((3,1))
    point_2D = project(bbox_center,
                       c2w,
                       P)
    point_2D = point_2D.reshape((-1,)) / point_2D[2,0]
    x, y, _ = point_2D
    x, y = int(x), int(y)
    radius = 30
    im_draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='red')
    return im


def align_mesh(mesh_points,
               center,
               bbox_center,
               mask_mat,
               c2w,
               P,
               default_sc=1):
    '''
    mesh_points: An array with shape of [3, N].
    Returns: An array with shape of [3, N].
    Make the projected mesh image's x_mean equal to the bbox_center.
    '''    
    bbox_center = np.array(bbox_center).reshape((3,1))

    bbox_center_2D = project(bbox_center, c2w, P)   # shape: [3, 1].
    mesh_points_2D = project(mesh_points, c2w, P)   # shape: [3, N].
    # mean_2D = np.mean(mesh_points_2D, axis=1, keepdims=True)
    mean_2D = (np.max(mesh_points_2D, axis=1, keepdims=True) + np.min(mesh_points_2D, axis=1, keepdims=True)) / 2
    bias_2D = bbox_center_2D - mean_2D
    bias_2D[1, 0] = 0                               # Only focus on x-axis.

    # Bias.
    bias_3D = inv_project(bias_2D, c2w, P)
    bias_3D = bias_3D - c2w[:3, 3:]
    mesh_points = mesh_points + bias_3D
    center = center + bias_3D

    # Compute scale.
    mesh_points_2D = mesh_points_2D / mesh_points_2D[2, :]
    mesh_width = np.max(mesh_points_2D[0, :]) - np.min(mesh_points_2D[0, :])
    _, mask_j_array, _ = np.where(mask_mat > 0)
    if mask_j_array.size == 0:
        return mesh_points, center
    elif np.max(mask_j_array) == (mask_mat.shape[1] - 1) or np.min(mask_j_array) == 0:
        # Mask maybe gets out the boundary.
        mask_width = mesh_width * default_sc     # Default scale.
    else:
        mask_width = np.max(mask_j_array) - np.min(mask_j_array)
    mesh_sc = mask_width / mesh_width

    # Scale.
    mesh_points = (mesh_points - center) * mesh_sc + center

    # import pdb; pdb.set_trace()
    return mesh_points, center, mesh_sc
