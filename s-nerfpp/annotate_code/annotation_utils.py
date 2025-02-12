import os
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

def read_calib(dir, idx):
    file = np.load(os.path.join(dir, 'target_poses.npy'))
    return file[idx, ...]


def read_P(dir, idx):
    file = np.load(os.path.join(dir, 'intrinsic.npy'))
    if len(file.shape) == 2:
        return file
    return file[idx, ...]


def read_bev_results(dir):
    file = np.load(os.path.join(dir, 'bev_results.npy'), allow_pickle=True)
    return file.item()


def read_bbox_txt_old_version(dir, idx):
    bbox_dir = os.path.join(dir, 'bbox')
    with open(os.path.join(bbox_dir, "%05d.txt" % idx), 'r') as f:
        _bbox_rec_list = f.readlines()
        bbox_rec_list = []
        for _bbox_rec in _bbox_rec_list:
            bbox_rec = {}
            _bbox_rec = _bbox_rec[:-1]    # del '\n'.
            _bbox_rec = _bbox_rec.split(' ')
            
            bbox_rec['track_id'] = int(_bbox_rec[0])
            bbox_rec['height'], bbox_rec['width'], bbox_rec['length'] = \
                float(_bbox_rec[1]), float(_bbox_rec[2]), float(_bbox_rec[3])
            bbox_rec['pos_x'], bbox_rec['pos_y'], bbox_rec['pos_z'] = \
                float(_bbox_rec[4]), float(_bbox_rec[5]), float(_bbox_rec[6])
            bbox_rec['rot_z'] = float(_bbox_rec[7])
            
            bbox_rec_list.append(bbox_rec)    
    
    return bbox_rec_list     


def read_bbox_txt(dir, idx):
    bbox_dir = os.path.join(dir, 'bbox')
    with open(os.path.join(bbox_dir, "%05d.txt" % idx), 'r') as f:
        _bbox_rec_list = f.readlines()
        bbox_rec_list = []
        for _bbox_rec in _bbox_rec_list:
            bbox_rec = {}
            _bbox_rec = _bbox_rec[:-1]    # del '\n'.
            _bbox_rec = _bbox_rec.split(' ')
            
            #################### Here to read the data you want. ####################
            bbox_rec['height'], bbox_rec['width'], bbox_rec['length'] = \
                float(_bbox_rec[8]), float(_bbox_rec[9]), float(_bbox_rec[10])
            bbox_rec['pos_x'], bbox_rec['pos_y'], bbox_rec['pos_z'] = \
                float(_bbox_rec[11]), float(_bbox_rec[12]), float(_bbox_rec[13])
            bbox_rec['rot_y'] = float(_bbox_rec[14])
            bbox_rec['category'] = _bbox_rec[0]
            #################### Here to read the data you want. ####################     
            bbox_rec_list.append(bbox_rec)    
    
    return bbox_rec_list     


def cvt_world_rec_to_world_corner(bbox_rec):
    '''
    Get 8 corners from bbox_rec (under world_coord).
    Note that 8 corners are in world_coord.
    '''
    l, w, h = bbox_rec['length'], bbox_rec['width'], bbox_rec['height']
    rot_z = bbox_rec['rot_z']
    x, y, z = bbox_rec['pos_x'], bbox_rec['pos_y'], bbox_rec['pos_z']
    
    rot = R.from_euler('z', rot_z, degrees=True).as_matrix()
    trans = np.array([x, y, z]).reshape((3,1))
    
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]                           # 1-----0     5-----4
    z_corners = [0,0,0,0,h,h,h,h]                                               # |     |     |     |
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]                           # |     |     |     |
    corners = np.vstack([x_corners, y_corners, z_corners])                      # |     |     |     |
                                                                                # 2-----3     6-----7
    corners = rot @ corners + trans
    return corners


def cvt_cam_rec_to_cam_corner(bbox_rec):
    '''
    Get 8 corners from bbox_rec (under cam_coord)
    Note that 8 corners are in cam_coord.
    '''
    l, w, h = bbox_rec['length'], bbox_rec['width'], bbox_rec['height']
    rot_y = bbox_rec['rot_y']
    x, y, z = bbox_rec['pos_x'], bbox_rec['pos_y'], bbox_rec['pos_z']    
    
    rot = R.from_euler('y', rot_y, degrees=False).as_matrix()
    trans = np.array([x, y, z]).reshape((3,1))
    
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]    
    corners = np.vstack([x_corners, y_corners, z_corners])
    
    corners = rot @ corners + trans
    return corners


def project(points_3D,
            c2w,
            P,
            is_cam_coord=True):
    '''
    3D to 2D.
    Returns:
        An array with shape of [3, N].
    '''
    world_coords = np.concatenate((points_3D, np.ones((1, points_3D.shape[1]))), axis=0)
    cam_coords = np.linalg.inv(c2w) @ world_coords if not is_cam_coord else world_coords
    plane_coords = P @ cam_coords[:3, ...]
    return plane_coords


def draw_one_line_from_points(points_2D, idx1, idx2, draw, category):
    
    x_from, y_from = points_2D[:, idx1].astype(np.int16)
    x_to, y_to = points_2D[:, idx2].astype(np.int16)
    
    fill_dict = {"Car": 'red', "Bicycle": 'blue', "Motorcycle": 'blue', "Object": 'yellow', "Pedestrian": 'green'}
    
    draw.line([(x_from, y_from), (x_to, y_to)], fill=fill_dict[category], width=3)
    

def cvt_cam_corner_to_world_corner(corners, c2w):
    
    corners = np.concatenate([corners, np.ones((1, corners.shape[1]))], axis=0)
    corners = c2w @ corners
    return corners[:3, :]


def cvt_world_corner_to_bev_corner(corners, bev_results):
    
    x_scale, y_scale = bev_results["x_scale"], bev_results["y_scale"]
    x_ori_bias, y_ori_bias = bev_results["x_ori_bias"], bev_results["y_ori_bias"]
    
    corners = corners[:2, :]
    corners[0, :] -= x_ori_bias; corners[1, :] -= y_ori_bias
    corners[0, :] *= x_scale;    corners[1, :] *= y_scale
    
    return corners


def visualize_one_rec(dir, idx):
    '''
    Returns an PIL.Image instance.
    '''
    # Read the data.
    c2w, P = read_calib(dir, idx), read_P(dir, idx)
    bbox_rec_list = read_bbox_txt(dir, idx)
    
    instance_n = len(bbox_rec_list)
    vis_im = Image.open(os.path.join(dir, 'image', "%05d.png" % idx))
    draw = ImageDraw.Draw(vis_im)
    
    bev_results = read_bev_results(dir)
    bev_im_mat = bev_results["bev_map_refined"][:,:,None].repeat(3, axis=-1)
    bev_im = Image.fromarray(bev_im_mat)
    bev_draw = ImageDraw.Draw(bev_im)
    
    if instance_n == 0: return vis_im, bev_im
    
    for instance_idx in range(instance_n):
        
        bbox_rec = bbox_rec_list[instance_idx]
        category = bbox_rec["category"]
        corners = cvt_cam_rec_to_cam_corner(bbox_rec)    
        corners_2D = project(corners, c2w, P)
        corners_2D = corners_2D[:2, ...] / corners_2D[2, ...]
        corners_3D_world = cvt_cam_corner_to_world_corner(corners, c2w)
        corners_3D_world = cvt_world_corner_to_bev_corner(corners_3D_world, bev_results)
        
        # Draw the lines.
        draw_one_line_from_points(corners_2D, 0, 1, draw, category)
        draw_one_line_from_points(corners_2D, 3, 2, draw, category)
        draw_one_line_from_points(corners_2D, 4, 5, draw, category)
        draw_one_line_from_points(corners_2D, 7, 6, draw, category)
        draw_one_line_from_points(corners_2D, 0, 3, draw, category)
        draw_one_line_from_points(corners_2D, 1, 2, draw, category)
        draw_one_line_from_points(corners_2D, 4, 7, draw, category)
        draw_one_line_from_points(corners_2D, 5, 6, draw, category)
        draw_one_line_from_points(corners_2D, 0, 4, draw, category)
        draw_one_line_from_points(corners_2D, 1, 5, draw, category)
        draw_one_line_from_points(corners_2D, 2, 6, draw, category)
        draw_one_line_from_points(corners_2D, 3, 7, draw, category)
        
        draw_one_line_from_points(corners_3D_world, 0, 1, bev_draw, category)
        draw_one_line_from_points(corners_3D_world, 0, 3, bev_draw, category)
        draw_one_line_from_points(corners_3D_world, 1, 2, bev_draw, category)
        draw_one_line_from_points(corners_3D_world, 3, 2, bev_draw, category)
    
    return vis_im, bev_im