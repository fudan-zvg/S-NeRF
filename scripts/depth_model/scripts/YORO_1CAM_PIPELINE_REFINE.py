import os
import numpy as np
import torch
import argparse
from os.path import join, basename, exists
import glob
import skimage.io as io
import sys
from PIL import Image
from timeit import default_timer as timer
import json

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from functools import reduce

from tools import transform_points, mask_outside_points, accumulate_points
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import LidarPointCloud

from copy import deepcopy
np.random.seed(1)


sepflow_path = join(os.path.dirname(__file__), '..', 'external', 'SeparableFlow-main', 'core')
lib_path = join(os.path.dirname(__file__), '..', 'external', 'SeparableFlow-main', 'libs')
root_path = join(os.path.dirname(__file__), '..', 'external', 'SeparableFlow-main')
if sepflow_path not in sys.path:
    sys.path.insert(0, sepflow_path)
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from sepflow import SepFlow
from utils.utils import InputPadder


DEVICE = 'cuda'
def load_image(imfile):
    if not os.path.exists(imfile):
        return None
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

 
def get_intrinsic_matrix(nusc, cam_token):       
    cam_data = nusc.get('sample_data', cam_token)
    cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    
    return np.array( cs_rec['camera_intrinsic'] )

    
def current_2_ref_matrix(nusc, current_sd_token, ref_sd_token):
    '''
    inputs:
        current_sd_token: current image token
        ref_sd_token: reference image token      
    '''    
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)    
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])

    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)    
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)           
    M_ref_from_global = reduce(np.dot, [ref_from_car, car_from_global])
    
    current_sd_rec = nusc.get('sample_data', current_sd_token)
    current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
    global_from_car = transform_matrix(current_pose_rec['translation'],
                                       Quaternion(current_pose_rec['rotation']), inverse=False)
    
    current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
    car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                        inverse=False)
    
    M_global_from_current = reduce(np.dot, [global_from_car, car_from_current])    
    M_ref_from_current = reduce(np.dot, [M_ref_from_global, M_global_from_current])
    
    return M_ref_from_current


def map_pointcloud_to_image(nusc, camera_token, height, accumulate, channel, min_dist): 
    
    # cam_record = nusc.get('sample', cam_sample_token)
    # points_token = cam_record['data']['LIDAR_TOP']
    # camera_token = cam_record['data']['CAM_FRONT']
    # Read the data.
    cam_sample_rec = nusc.get('sample', nusc.get('sample_data', camera_token)['sample_token'])
    points_token = cam_sample_rec['data']['LIDAR_TOP']
    pointsensor = nusc.get('sample_data', points_token)
    cam = nusc.get('sample_data', camera_token)
    
    time_dist = pointsensor['timestamp'] - cam['timestamp']
    while True:
        if time_dist == 0:
            break
        elif time_dist > 0:
            if not pointsensor['prev']:
                break
            pointsensor = nusc.get('sample_data', pointsensor['prev'])
            new_dist = pointsensor['timestamp'] - cam['timestamp']
            if new_dist <= 0:
                pointsensor = nusc.get('sample_data', pointsensor['next']) if abs(new_dist) > abs(time_dist) else pointsensor
                break
            else:
                time_dist = new_dist
        elif time_dist < 0:
            if not pointsensor['next']:
                break
            pointsensor = nusc.get('sample_data', pointsensor['next'])
            new_dist = pointsensor['timestamp'] - cam['timestamp']
            if new_dist >= 0:
                pointsensor = nusc.get('sample_data', pointsensor['prev']) if abs(new_dist) > abs(time_dist) else pointsensor
                break
            else:
                time_dist = new_dist
    
    
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])    
    pcl_path = join(nusc.dataroot, pointsensor['filename'])
    pc = LidarPointCloud.from_file(pcl_path)

    # Transform the points.
    depths = transform_points(nusc, cam, pointsensor, pc)
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)[:2, :]
    points, depths, valid = mask_outside_points(points, depths, height) 
    pc.points = pc.points[:, valid]   
    
    # If need to accumulate.
    if accumulate:
        pointsensor_prev = pointsensor
        pointsensor_next = pointsensor
        for i in range(accumulate):
            
            # Read the prev pointcloud.
            pointsensor_prev, pc_prev = accumulate_points(nusc, pointsensor_prev, "prev")
            
            # Mask the points on the collecting car.
            cam_prev = nusc.get('sample_data', nusc.get('sample', pointsensor_prev['sample_token'])['data'][channel])
            temp_pc_prev = deepcopy(pc_prev)
            mask = transform_points(nusc, cam_prev, pointsensor_prev, temp_pc_prev) > min_dist
            pc_prev.points = pc_prev.points[:, mask]
            
            # Transform the points, map and mask the points.
            depths_prev = transform_points(nusc, cam, pointsensor_prev, pc_prev)  
            points_prev = view_points(pc_prev.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)[:2, :]
            points_prev, depths_prev, valid_prev = mask_outside_points(points_prev, depths_prev, height)
            pc_prev.points = pc_prev.points[:, valid_prev]
                
            points = np.hstack((points_prev, points))
            depths = np.hstack((depths_prev, depths))
            pc.points = np.hstack((pc_prev.points, pc.points))
                        
            # The same for the next pointcloud.
            pointsensor_next, pc_next = accumulate_points(nusc, pointsensor_next, "next")
            cam_next = nusc.get('sample_data', nusc.get('sample', pointsensor_next['sample_token'])['data'][channel])
            temp_pc_next = deepcopy(pc_next)
            mask = transform_points(nusc, cam_next, pointsensor_next, temp_pc_next) > min_dist
            pc_next.points = pc_next.points[:, mask]
            depths_next = transform_points(nusc, cam, pointsensor_next, pc_next)  
            points_next = view_points(pc_next.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)[:2, :]
            points_next, depths_next, valid_next = mask_outside_points(points_next, depths_next, height)
            pc_next.points = pc_next.points[:, valid_next]
                
            points = np.hstack((points, points_next))
            depths = np.hstack((depths, depths_next))
            pc.points = np.hstack((pc.points, pc_next.points))
    
    points, depths, valid = mask_outside_points(points, depths, height)
    points = np.vstack((points, depths))
    pc.points = pc.points[:, valid]

    return points, pc    # [3, N] array.


def proj_2_next(nusc, pc, camera_token):
    
    cam_rec = nusc.get('sample_data', camera_token)
    prev_cam_rec = nusc.get('sample_data', cam_rec['next'])
    
    cs_record = nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))
    
    poserecord = nusc.get('ego_pose', cam_rec['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)        
    pc.translate(np.array(poserecord['translation']))
    
    poserecord = nusc.get('ego_pose', prev_cam_rec['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T) 
    
    cs_record = nusc.get('calibrated_sensor', prev_cam_rec['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)     
    
    depths = pc.points[2, :]    
    
    # Map.
    next_points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)[:2, :]          

    return next_points


def proj_2_prev(nusc, pc, camera_token):
    
    cam_rec = nusc.get('sample_data', camera_token)
    prev_cam_rec = nusc.get('sample_data', cam_rec['prev'])
    
    cs_record = nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))
    
    poserecord = nusc.get('ego_pose', cam_rec['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)        
    pc.translate(np.array(poserecord['translation']))
    
    poserecord = nusc.get('ego_pose', prev_cam_rec['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T) 
    
    cs_record = nusc.get('calibrated_sensor', prev_cam_rec['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)     
    
    depths = pc.points[2, :]    
    
    # Map.
    next_points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)[:2, :]          

    return next_points


def consistency_check(points, next_points, im_flow, min_depth, max_depth, base_thres):
    # points: 3*N
    # next_points: 2*N
    points_n = points.shape[1]
    mask = np.ones(points_n, dtype=bool)
    for i in range(points_n):
        x1, y1 = points[0, i], points[1, i]
        x2, y2 = next_points[0, i], next_points[1, i]
        x1_one, y1_one = round(x1), round(y1)
        
        lidar_flow = np.array([x2-x1, y2-y1])
        flow = im_flow[y1_one, x1_one]
        
        depth = points[2, i]
        flow_norm = np.linalg.norm(flow)
        thres = 3 + flow_norm * base_thres
        if flow_norm > 50 and flow_norm < 100:
            thres = flow_norm * .3 + 5
        elif flow_norm > 100 and flow_norm < 150:
            thres = flow_norm * .8 + 5
        elif flow_norm > 150 or depth < 3.5:
            thres = 10000000
        if np.linalg.norm(flow - lidar_flow) > thres:
            mask[i] = False
        
    return mask

def points2im(points):
    h, w = 900, 1600
    im = np.zeros((h, w), dtype=np.uint16)
    points_n = points.shape[1]
    for i in range(points_n):
        x, y = round(points[0, i]), round(points[1, i])
        im[y, x] = round(points[2, i] * 256)
    return im

def im2points(im):
    im = torch.tensor(im)
    h, w = im.size()
    x = torch.arange(w).repeat(h)
    y = torch.arange(h).repeat([w,1]).T.flatten()
    points = torch.stack((x,y,(im/256).view((-1,))), axis=0)
    return points 

def consistency_check_new(points, another_points, flow_im):
    d_lidar = another_points - points
    flow_im = flow_im[points[1, :].long(), points[0, :].long()]
    confidence = torch.zeros((900, 1600), dtype=float)
    confidence[points[1, :].long(), points[0, :].long()] = torch.norm(d_lidar - flow_im.T, dim=0) / (torch.norm(flow_im.T, dim=0))
    confidence = confidence.cpu().numpy()
    mask = np.where(confidence > .25)
    return mask  

def two_D_2_three_D(points, camera_intrinsic):
    '''
    2D to 3D.
    '''
    n = points.size()[1]
    c_u = camera_intrinsic[0, 2]
    c_v = camera_intrinsic[1, 2]
    f_u = camera_intrinsic[0, 0]
    f_v = camera_intrinsic[1, 1]
    # b_x = camera_intrinsic[0, 3] / -f_u
    # b_y = camera_intrinsic[1, 3] / -f_v
    b_x = 0
    b_y = 0
    
    points_T = points.clone().T
    x = ((points_T[:, 0] - c_u) * points_T[:, 2]) / f_u + b_x
    y = ((points_T[:, 1] - c_v) * points_T[:, 2]) / f_v + b_y
    pts_3d_rect = np.zeros((n, 3))
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = points_T[:, 2]  
    pts_3d_rect = np.hstack((pts_3d_rect, np.ones((n, 1))))  
    
    return LidarPointCloud(pts_3d_rect.T)


def proj_2_another(nusc, pc, camera_token, mode):
    
    cam_rec = nusc.get('sample_data', camera_token)
    another_cam_rec = nusc.get('sample_data', cam_rec[mode])
    
    cs_record = nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))
    
    poserecord = nusc.get('ego_pose', cam_rec['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)        
    pc.translate(np.array(poserecord['translation']))
    
    poserecord = nusc.get('ego_pose', another_cam_rec['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T) 
    
    cs_record = nusc.get('calibrated_sensor', another_cam_rec['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)     
    
    depths = pc.points[2, :]    
    
    # Map.
    next_points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)[:2, :]          

    return torch.tensor(next_points)


def one_cam_process(args):
    channel = args.channel
    if args.dir_data == None:
        this_dir = os.path.abspath(os.path.dirname(__file__))
        args.dir_data = os.path.join(this_dir, '..', 'temp')
        os.makedirs(args.dir_data, exist_ok=True)

    data_root = '../../data/'
    dir_nuscenes = os.path.join(data_root, f'nuScenes/{args.version.split("-")[-1]}/')
    nusc = NuScenes(version=args.version, dataroot = dir_nuscenes, verbose=False)
    
    # Part 1. Get Sweeps Tokens.
    scene_name = args.scene_name
    channel_tokens_file = os.path.join(data_root, 'scenes', scene_name, 'channel_tokens.json')
    with open(channel_tokens_file, 'r') as f:
        channel_tokens = json.load(f)
    sweeps_tokens = channel_tokens[channel]

    # Part 2.
    dir_data_out = join(args.dir_data, '6cam_data', channel)
    flow_data_out = join(args.dir_data, 'flow_data')
    if not os.path.exists(dir_data_out):
        os.makedirs(dir_data_out)
    if not os.path.exists(flow_data_out):
        os.makedirs(flow_data_out)

    'remove all files in the output folder'
    f_list=glob.glob(join(dir_data_out,'*'))
    for f in f_list:
        os.remove(f)
    print('removed %d old files in output folder' % len(f_list))

    ct = 0         
    for sweeps_token in sweeps_tokens:  
        cam_token = sweeps_token
        cam_data = nusc.get('sample_data', cam_token)
        if cam_data['next'] and cam_data['prev']:
            cam_path1 = join(nusc.dataroot, cam_data['filename'])
            im1 = io.imread(cam_path1)
            
            cam_token2 = cam_data['next']        
            cam_data2 = nusc.get('sample_data', cam_token2)
            cam_path2 = join(nusc.dataroot, cam_data2['filename'])
            im2 = io.imread(cam_path2)
            
            cam_token_prev = cam_data['prev']
            cam_data0 = nusc.get('sample_data', cam_token_prev)
            cam_path0 = join(nusc.dataroot, cam_data0['filename'])
            im0 = io.imread(cam_path0)
            
            io.imsave(join(dir_data_out, '%05d_im.jpg' % (ct)), im1)
            io.imsave(join(dir_data_out, '%05d_im_next.jpg' % (ct)), im2)
            io.imsave(join(dir_data_out, '%05d_im_prev.jpg' % (ct)), im0)
        
        #################### Modified on 2023/1/11 #####################            
        else:
            cam_path1 = join(nusc.dataroot, cam_data['filename'])
            im1 = io.imread(cam_path1)            
            io.imsave(join(dir_data_out, '%05d_im.jpg' % (ct)), im1)
            
            if cam_data['next']:
                cam_token2 = cam_data['next']        
                cam_data2 = nusc.get('sample_data', cam_token2)
                cam_path2 = join(nusc.dataroot, cam_data2['filename'])
                im2 = io.imread(cam_path2)                
                io.imsave(join(dir_data_out, '%05d_im_next.jpg' % (ct)), im2)
                
            if cam_data['prev']:
                cam_token_prev = cam_data['prev']
                cam_data0 = nusc.get('sample_data', cam_token_prev)
                cam_path0 = join(nusc.dataroot, cam_data0['filename'])
                im0 = io.imread(cam_path0)
                io.imsave(join(dir_data_out, '%05d_im_prev.jpg' % (ct)), im0)
        
        #################### Modified on 2023/1/11 #####################            
            
        ct += 1
        print('Save image %d/%d' % ( ct, len(sweeps_tokens[:-1]) ) )
        

    # Part 3.
    if args.model == None:
        this_dir = os.path.abspath(os.path.dirname(__file__))
        args.model = join(this_dir, '..', 'external', 'SeparableFlow-main', 'models', 'sepflow_universal.pth')  #######

        
    out_dir = join(args.dir_data, '6cam_data', channel)
    flow_dir = join(args.dir_data, 'flow_data', channel)
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)
    f_list=glob.glob(join(flow_dir,'*'))
    for f in f_list:
        os.remove(f)
    print('removed %d old files in output folder' % len(f_list))    
        
    model = torch.nn.DataParallel(SepFlow(args))
    msg = model.load_state_dict(torch.load(args.model)['state_dict'], strict=False)
    print(msg)

    model = model.module
    model.cuda()
    model.eval()
        
    im_list = np.array(np.sort(glob.glob(join(out_dir, '*im.jpg'))))

    N = len(im_list)
        
    print('Total sample number:', N)
        
    ct = 0
    for sample_idx in range(0, N):
        
        f_im1 = im_list[sample_idx]
        
        im1 = load_image(f_im1)
        
        f_im_next = f_im1[:-4] + '_next.jpg'
        f_im_prev = f_im1[:-4] + '_prev.jpg'

        im2 = load_image(f_im_next)
        im0 = load_image(f_im_prev)
        
        padder = InputPadder(im1.shape, mode='others')        
                       
        if im2 is not None:                                               
            im1, im2 = padder.pad(im1, im2)
            if channel in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'] or im0 is None:
                with torch.no_grad():
                    flow_low, flow_up = model(im1, im2, iters=20)
                    flow_up = padder.unpad(flow_up)
                    flow = flow_up[0].permute(1,2,0).cpu().numpy()
                    path_flow = f_im1[:-6] + 'flow_next.npy'
                    np.save(path_flow, flow)
                    np.save(join(flow_dir, basename(f_im1)[:-6] + 'flow_next.npy'), flow)
            
            im1 = load_image(f_im1)
            padder = InputPadder(im1.shape, mode='others')
        
        if im0 is not None:
        
            im1, im0 = padder.pad(im1, im0)
            
            if channel in ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'] or im2 is None:
            # if True:
                with torch.no_grad():
                    flow_low, flow_up = model(im1, im0, iters=20)
                    flow_up = padder.unpad(flow_up)
                    flow = flow_up[0].permute(1,2,0).cpu().numpy()
                    
                    path_flow = f_im1[:-6] + 'flow_prev.npy'
                    
                    np.save(path_flow, flow)
                    np.save(join(flow_dir, basename(f_im1)[:-6] + 'flow_prev.npy'), flow)
        
        ct += 1
        print('compute flow %d/%d' % ( ct, N ) )
        
    del model
    del flow_up
    del flow_low
    del padder
    torch.cuda.empty_cache()
        

    # Part 4.
    dir_data_out = join(args.dir_data, '6cam_data', channel)

    ct = 0         
    for sweeps_token in sweeps_tokens:  
        cam_token = sweeps_token
        cam_data = nusc.get('sample_data', cam_token)
        
        if cam_data['next']:
                
            cam_token1 = cam_data['next']                           
            
            K = get_intrinsic_matrix(nusc, cam_token)
            T = current_2_ref_matrix(nusc, cam_token, cam_token1)
        
        #################### Modified on 2023/1/11 #####################                        
        else:
            
            cam_token1 = cam_data['prev']    
            
            K = get_intrinsic_matrix(nusc, cam_token)
            T = current_2_ref_matrix(nusc, cam_token, cam_token1)            
        #################### Modified on 2023/1/11 #####################                
        
        np.savez(join(dir_data_out, '%05d_matrix.npz' % (ct)), K=K, T=T)    
            
        ct += 1
        print('Save matrix %d/%d' % ( ct, len(sweeps_tokens[:-1]) ) )

    # Part 5.
    N_total = len(sweeps_tokens)
    print('Total sample number:', N_total)
    ct = 0
    with open(join(dir_data_out, 'points_n.txt'), 'w') as f:
        running_mean = 0
        for sweeps_token in sweeps_tokens:
            start = timer()
            #################### Modified on 2023/1/11 #####################                
            if channel in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
                mode = 'next'
                flow_f_name = join(dir_data_out, '%05d_flow_next.npy' % (ct))
                if os.path.exists(flow_f_name):
                    im_flow = np.load(flow_f_name)
                else:
                    mode = 'prev'
                    flow_f_name = join(dir_data_out, '%05d_flow_prev.npy' % (ct))
                    im_flow = np.load(flow_f_name)
            elif channel in ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                mode = "prev"
                flow_f_name = join(dir_data_out, '%05d_flow_prev.npy' % (ct))
                if os.path.exists(flow_f_name):
                    im_flow = np.load(flow_f_name)
                else:
                    mode = "next"
                    flow_f_name = join(dir_data_out, '%05d_flow_next.npy' % (ct))
                    im_flow = np.load(flow_f_name)
            
            #################### Modified on 2023/1/11 #####################                        
                    
            # matrix = np.load(join(dir_data_out, '%05d_matrix.npz' % (ct)))
            # seg = np.load(join(dir_data_out, '%05d_seg.npy' % ct))
            # K = matrix['K']
                        
            # Accumualte.
            if channel in ['CAM_FRONT']:
                min_dist = 2.5
            elif channel in ['CAM_BACK']:
                min_dist = 6
            else:
                min_dist = 1.5      
            
            if channel in ['CAM_FRONT', 'CAM_BACK']:
                frames = 12
            else:
                frames = 6
            current_points, current_pc = map_pointcloud_to_image(nusc, sweeps_token, 900, frames, channel, min_dist)
            cam_rec = nusc.get('sample_data', sweeps_token)
            # print(cam_rec['timestamp'])
            camera_intrinsic = np.array(nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])['camera_intrinsic'])
            # pc = two_D_2_three_D(current_points, camera_intrinsic)
            if mode == 'prev':
                another_points = proj_2_prev(nusc, current_pc, sweeps_token)
                print('use prev flow')
            else:
                another_points = proj_2_next(nusc, current_pc, sweeps_token)
                print('use next flow')
                    
            max_depth = np.max(current_points[2, :])
            min_depth = np.min(current_points[2, :])
            
            base_thres = .1
            mask = consistency_check(current_points, another_points, im_flow, min_depth, max_depth, base_thres)
            points_n = np.sum(mask)
            if ct == 0:
                pass
            else:
                try_ct = 0
                while try_ct < 2 and points_n < running_mean and (running_mean - points_n) > .2 * running_mean:
                    base_thres += .05
                    mask = consistency_check(current_points, another_points, im_flow, min_depth, max_depth, base_thres)
                    points_n = np.sum(mask)  
                    try_ct += 1
            running_mean = (running_mean * ct + points_n) / (ct + 1)
                              
            print(np.sum(mask))            
            current_points = current_points[:, mask]    

            gt = points2im(current_points)        
        
            
            Image.fromarray(gt).save(join(dir_data_out, '%05d_gtim.png' % (ct)))
                    
            ct += 1
            print(f'compute depth {ct}/{len(sweeps_tokens)}')
            
            end = timer()
            t = end-start     
            print('Time used: %.1f s' % t)
            
            f.write(str(ct) + ':  ' + str(np.sum(mask)))
            f.write('\n')
            
            
    # Part 6.
    data_dir = join(args.dir_data, '6cam_data', channel)
    channel_dir = join(args.dir_data, 'prepared_sweeps_data', channel)
    if not exists(channel_dir):
        os.makedirs(channel_dir)
    im_list = np.array(np.sort(glob.glob(join(data_dir, '*im.jpg'))))
    depth_list = np.array(np.sort(glob.glob(join(data_dir, '*gtim.png'))))
    calib_list = np.array(np.sort(glob.glob(join(data_dir, '*.npz'))))
    
    out_dir = join(args.dir_data, 'prepared_sweeps_data', channel, 'img')
    if not exists(out_dir):
        os.makedirs(out_dir)
    f_list=glob.glob(join(out_dir,'*'))
    for f in f_list:
        os.remove(f)
    print('removed %d old files in output folder' % len(f_list))
        
    for fn in im_list:
        im_file = Image.open(fn)
        im_file.save(join(out_dir, basename(fn)[:-7] + '.png'))
    
    out_dir = join(args.dir_data, 'prepared_sweeps_data', channel, 'gt')
    if not exists(out_dir):
        os.makedirs(out_dir)
    f_list=glob.glob(join(out_dir,'*'))
    for f in f_list:
        os.remove(f)
    print('removed %d old files in output folder' % len(f_list))    
    for fn in depth_list:        
        gt_file = Image.open(fn)
        gt_mat = np.array(gt_file).astype(np.uint16)
        gt_mat = gt_mat   # Scale.
        Image.fromarray(gt_mat).save(join(out_dir, basename(fn)[:-9] + '.png'))   
    
    out_dir = join(args.dir_data, 'prepared_sweeps_data', channel, 'calib')
    if not exists(out_dir):
        os.makedirs(out_dir)    
    f_list=glob.glob(join(out_dir,'*'))
    for f in f_list:
        os.remove(f)
    print('removed %d old files in output folder' % len(f_list))    
    for fn in calib_list:
        file = np.load(fn)
        K = file['K']
        calib_str = ''
        for num in K.ravel():
            calib_str += (str(num)) + ' '
        calib_str = calib_str[:-1]         
        
        outpath = join(out_dir, basename(fn)[:-11] + '.txt')
        if exists(outpath):
            continue        
        f = open(outpath, 'w')
        f.write(calib_str)
        f.write('\n')
        f.close() 
    
    idx_list = list(range(0, len(sweeps_tokens)))
    out_dir = join(args.dir_data, 'prepared_sweeps_data', channel, 'json')
    if not exists(out_dir):
        os.makedirs(out_dir)
    out_dir = join(args.dir_data, 'prepared_sweeps_data', channel, 'json', 'data_test.json')
    with open(out_dir, 'w') as f:
        total_dict = {"train":[], "val":[], "test":[]}
        for idx in idx_list:
            one_dict = {"rgb": join('prepared_sweeps_data', channel, 'img', '%05d.png' % idx),
                        "depth": join('prepared_sweeps_data', channel, 'gt', '%05d.png' % idx),
                        "gt": join('prepared_sweeps_data', channel, 'gt', '%05d.png' % idx),
                        "K": join('prepared_sweeps_data', channel, 'calib', '%05d.txt' % idx)}
            total_dict['test'].append(one_dict)
        json.dump(total_dict, f)    
        
    
    # Part 7.
    if args.depth_completion_mode == "NLSPN":
        NLSPN_path = join(this_dir, '..', 'external', 'NLSPN_ECCV20/src/')
        # os.chdir('/var/lib/docker/data/users/liwenye/NLSPN_ECCV20/src/')
        os.chdir(NLSPN_path)
        depth_out_dir = os.path.abspath(join('..', '..', '..', 'temp', '6cam_data', channel))
        os.system('python main.py --dir_data ../../../temp/\
            --data_name KITTIDC --split_json ../../../temp/prepared_sweeps_data/'+ channel +'/json/data_test.json\
                --patch_height 900 --patch_width 1600 --gpus %s --max_depth 150.0 --num_sample 0 --batch_size 1 --test_only\
                    --pretrain ../results/model_00010.pt --save scene_' % args.gpu_for_NLSPN + channel +\
                        ' --save_image --save_result_only --preserve_input --channel ' + channel + ' --out_dir %s' % depth_out_dir)
    elif args.depth_completion_mode == "SDC":
        SDC_path = join(this_dir, '..', 'external', 'Sparse-Depth-Completion')
        # os.chdir('/var/lib/docker/data/users/liwenye/Sparse-Depth-Completion')
        os.chdir(SDC_path)
        save_path = join('Saved')
        data_path = join('..', '..', 'temp', 'prepared_sweeps_data', channel)
        SDC_out_dir = join('..', '..', 'temp', '6cam_data', channel)
        os.system('python Test/test.py --save_path %s --data_path %s --out_dir %s' % (save_path, data_path, SDC_out_dir))

    # Part 8.
    mseg_path = join(this_dir, '..', 'external', 'mseg-semantic')
    os.chdir(mseg_path)
    out_dir = os.path.abspath(join('mseg_semantic', 'data_seg', channel, 'gray'))
    f_list = glob.glob(join(out_dir,'*'))
    for f in f_list:
        os.remove(f)
    print('removed %d old files in output folder' % len(f_list))
    os.system('CUDA_VISIBLE_DEVICES=%s python -u mseg_semantic/tool/universal_demo.py\
        --config mseg_semantic/config/test/default_config_1080_ss.yaml\
        --file_save mseg_semantic/experiments model_name mseg-3m model_path mseg_semantic/checkpoints/mseg-3m.pth input_file\
            ../../temp/prepared_sweeps_data/%s/img/ save_folder mseg_semantic/data_seg/' % (args.gpu_for_NLSPN, channel) + channel + ' base_size 900')
    
    
    # Part 9.
    # script_path = join('..', 'rc-pda', 'extra_scripts')
    # os.chdir('/var/lib/docker/data/users/liwenye/rc-pda/extra_scripts/')
    os.chdir(this_dir)
    dir_data_out = join('..', 'temp', '6cam_data', channel)
    depth_root = join('..', 'output', '6cam_depth_data')
    depth_out = join(depth_root, channel)
    if not os.path.exists(depth_root):
        os.mkdir(depth_root)
    if not os.path.exists(depth_out):
        os.makedirs(depth_out)
    f_list=glob.glob(join(depth_out,'*'))
    for f in f_list:
        os.remove(f)
    print('removed %d old files in output folder' % len(f_list))    
    ct = 0
    if True:
        mask_fname_list = sorted(os.listdir(out_dir))
        for sweeps_token in sweeps_tokens:
            
            if channel in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
                mode = "next"
                if nusc.get('sample_data', sweeps_token)['next'] == '':
                    mode = "prev"
                
            elif channel in ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                mode = "prev"
                if nusc.get('sample_data', sweeps_token)['prev'] == '':
                    mode = "next"

            depth_im = np.array(Image.open(join(dir_data_out, '%010d.png' % ct)))
            mask_im = np.array(Image.open(join(out_dir, mask_fname_list[ct])))
            points_2D = im2points(depth_im)
            cam_token = sweeps_token
            cam_rec = nusc.get('sample_data', cam_token)
            cam_intrinsic = torch.tensor(nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])['camera_intrinsic'])
            pc_current = two_D_2_three_D(points_2D, cam_intrinsic)
            
            another_points_2D = proj_2_another(nusc, pc_current, cam_token, mode)
            
            if mode == 'next':
                im_flow = torch.tensor(np.load(join(dir_data_out, '%05d_flow_next.npy' % (ct))))
            elif mode == 'prev':
                im_flow = torch.tensor(np.load(join(dir_data_out, '%05d_flow_prev.npy' % (ct))))
            else:
                im_flow = torch.tensor(np.load(join(dir_data_out, '%05d_flow.npy' % (ct))))
            mask = consistency_check_new(points_2D[:2, :], another_points_2D, im_flow)
            
            depth_im[mask] = 0
            depth_im = depth_im.astype(np.uint16)
            depth_im[mask_im == 142] = 200 * 256
            Image.fromarray(depth_im).save(join(depth_out, '%05d_finalim.png' % (ct)))
            
            ct += 1
            print(f'refine depth {ct}/{len(sweeps_tokens)}')
                     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--scene_name', type=str) 
    parser.add_argument('--model')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')   
    parser.add_argument('--channel', type=str)
    parser.add_argument('--gpu_for_NLSPN', type=str)
    parser.add_argument('--depth_completion_mode', type=str, default="NLSPN")
    args = parser.parse_args()    
    
    one_cam_process(args)