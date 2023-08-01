import numpy as np
import os.path as osp
import cv2
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image

MAX_WIDTH= 1600
MAX_HEIGHT = 900


def np2png_matrix(points, depths, height):
    '''
    Transform the numpy array ([2, k]) to the png_matrix ([m, n]).
    args:
        points: An [2, k] array, representing the points.
        depths: An [k] array, containing the depths.
        height: The height of the output matrix.
    returns:
        png_matrix: An [m, n] array.
    '''
    png_matrix = np.zeros((height, MAX_WIDTH))
    for i in range(points.shape[1]) :
        png_matrix[round(points[1, i]), round(points[0, i])] = round(depths[i] * 256)
    png_matrix = png_matrix.astype(np.uint16)     
    
    return png_matrix   


def png_matrix2np(png_matrix, real_depths):
    '''
    Transform the png_matrix ([m, n]) to the numpy array ([2, k]).
    args:
        png_matrix: An [m, n] array.
        real_depths: whether return the real depths or depths(i.e. pixel value).
    returns:
        points: An [2, k] array, representing the points.
        depths: An [k] array, containing the depths.
    '''
        # png_matrix to array.
    points_list = []
    depths_list = []
    for i in range(png_matrix.shape[1]):
        for j in range(png_matrix.shape[0]):
            if png_matrix[j, i]:
                points_list.append((i, j))
                depths_list.append(png_matrix[j, i])
    if points_list != []:
        points = np.array(points_list).T
    else:
        points = np.array(([], []))
    depths = np.array(depths_list)
    if real_depths:
        depths.astype(np.float32)
        depths = depths / 256
    
    return points, depths


def accumulate_points(nusc, pointsensor, type):
    '''
    help:
        accumulate another point cloud file.
    args:
        nusc: An instance of class 'nusc'.
        pointsensor: the record of points. (in table 'sample_data')
        type: "next" or "prev".
    return:
        pointsensor_xxxx: The "next" or "prev" of the pointsensor.
        pc_xxxx: The "next" or "prev" pointcloud.
    '''
    assert type in ("next", "prev"), "Invalid type. Type should be \"next\" or \"prev\"."
    if type == "next": 
        if not pointsensor['next']: 
            # No next file, return with no change.
            print("Warning: There is no more next pointcloud file.")
            pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
            pc = LidarPointCloud.from_file(pcl_path)
            return pointsensor, pc
        pointsensor_next = nusc.get('sample_data', pointsensor['next'])
        pcl_next_path = osp.join(nusc.dataroot, pointsensor_next['filename'])
        pc_next = LidarPointCloud.from_file(pcl_next_path) 
        return pointsensor_next, pc_next
    elif type == "prev":
        if not pointsensor['prev']:
            # No prev file, return with no change.
            print("Warning: There is no more prev pointcloud file.")
            pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
            pc = LidarPointCloud.from_file(pcl_path)
            return pointsensor, pc
        pointsensor_prev = nusc.get('sample_data', pointsensor['prev'])
        pcl_prev_path = osp.join(nusc.dataroot, pointsensor_prev['filename'])
        pc_prev = LidarPointCloud.from_file(pcl_prev_path)
        return pointsensor_prev, pc_prev            


def transform_points(nusc, cam, pointsensor, pc):
    '''
    help:
        transform points to the target camera.
    args:
        nusc: An instance of class 'nusc'.
        cam: the record of camera. (in table 'sample_data')
        pointsensor: the record of points. (in table 'sample_data')
        pc: A LidarPointCloud instance.
    return:
        depths
    Note:
        Calling this function will change the pc.points directly.
    '''
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    
    return depths


def mask_outside_points(points, depths, height):
    '''
    help:
        mask the points outside the image and change the axies.
    args:
        points: An [2, k] array.
        depths: An [k] array.
        height: the height of image.
    return:
        points and depths after masking.
    '''
    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1.0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < MAX_WIDTH - 1)
    mask = np.logical_and(mask, points[1, :] > MAX_HEIGHT - height + 1)
    mask = np.logical_and(mask, points[1, :] < MAX_HEIGHT - 1)
    points = points[:, mask]
    depths = depths[mask]
    
    # Change the axies.
    points[1, :] -= (MAX_HEIGHT - height)  
    
    return points, depths, mask


def visualize(points, depths, im):
    '''
    Plot the velodyne points in the im image.
    args:
        points: An [2, k] array, representing the points' location.
        depths: An array, representing the depths value.
        im: An Image instance.
    returns:
        im_mat: An Image instance.
    '''
    im_mat = np.array(im)
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_TURBO).reshape(256, 3))
    grey_to_color_map = dict(zip(gray_values, color_values))
    color_depths = depths / np.max(depths) * 255   # scale.
    for i in range(points.shape[1]):
        im_mat[round(points[1, i]), round(points[0, i])] = np.array(grey_to_color_map[round(color_depths[i])])
    return Image.fromarray(im_mat)


def visualize_depth(im_mat):
    '''
    Plot the velodyne points in the im image.
    args:
        im_mat: An numpy array.
    returns:
        im_mat: An Image instance.
    '''
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).reshape(256, 3))
    grey_to_color_map = dict(zip(gray_values, color_values))
    im_mat = im_mat / np.max(im_mat) * 255
    h, w = im_mat.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            result[i, j] = np.array(grey_to_color_map[round(im_mat[i, j])])
    return Image.fromarray(result)