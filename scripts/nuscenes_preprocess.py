from imageio import save
import numpy as np
import os
import cv2
import json

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default = '' )
    parser.add_argument('--total_num', type=int, default = 30, help = 'The frames needed')
    parser.add_argument('--camera_index', type=list, default = [0,1,2,3,4,5], help = 'Cameras chosen')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--first_sample_token', type=str, default='')
    parser.add_argument('--savedir', type=str, default='')

    args = parser.parse_args()

    # Data Initialization
    nusc = NuScenes(version='v1.0-mini',dataroot=args.dataroot, verbose=True)
    max_sweep = 10
    images = []
    ego2global_rts = []
    cam2ego_rts = []
    cam_intrinsics = []
    sensor_all = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']
    sensor = [sensor_all[i] for i in args.camera_index]

    temp_sample = nusc.get('sample', args.first_sample_token)
    i = 0
    while(i < 20):
        temp_sample = nusc.get('sample', temp_sample['next'])
        i += 1

    IDX = 0
    sample_idx_list = {}
    for s in sensor:
        temp_data = nusc.get('sample_data',temp_sample['data'][s])
        for i in range(args.total_num):        
            data_path , _ , cam_intrinsic = nusc.get_sample_data(temp_data['token'])
            if not os.path.exists(data_path):
                temp_data=nusc.get('sample_data',temp_data['next'])
                continue
            if(temp_data['is_key_frame']):
                sample_idx_list[IDX] = temp_data['token']
            IDX += 1

            cam_intrinsics.append(cam_intrinsic.astype(np.float32))
            #image
            fname = data_path
            img = cv2.imread(fname)
            images.append(img)
            #ego2global
            temp_ego2global=nusc.get('ego_pose',temp_data['ego_pose_token'])
            ego2global_r=Quaternion(temp_ego2global['rotation']).rotation_matrix
            ego2global_t=np.array(temp_ego2global['translation'])
            ego2global_rt=np.eye(4)
            ## correct the ego2global pose
            ego2global_rt[:3,:3]= ego2global_r
            ego2global_rt[:3,3]= ego2global_t
            
            ego2global_rts.append(ego2global_rt.astype(np.float32))
            temp_cam2ego=nusc.get('calibrated_sensor',temp_data['calibrated_sensor_token'])
            #cam2ego
            cam2ego_r=Quaternion(temp_cam2ego['rotation']).rotation_matrix
            cam2ego_t=np.array(temp_cam2ego['translation'])
            cam2ego_rt = np.eye(4)
            cam2ego_rt[:3, :3] = cam2ego_r
            cam2ego_rt[:3, 3] = cam2ego_t
            cam2ego_rts.append(cam2ego_rt.astype(np.float32))
            temp_data=nusc.get('sample_data',temp_data['next'])


    camtoworlds = [ego2global_rts[i] @cam2ego_rts[i] for i in range(len(cam2ego_rts))]
    camtoworlds = np.stack(camtoworlds,axis=0)
    center = camtoworlds[len(camtoworlds)//2,:3,3]
    trans_center = camtoworlds[len(camtoworlds)//2,:,:].copy()

    P = np.eye(3)
    c2w = camtoworlds.copy()
    c2w[:,:3,3] = camtoworlds[:,:3,3]-trans_center[:3,3]
    c2w[:,:3,3] = c2w[:,:3,3]@ trans_center[:3,:3] @ P
    c2w[:,:3,:3] = np.moveaxis(np.dot(np.linalg.inv(trans_center[:3,:3]@P),c2w[:,:3,:3] @ P),1,0)

    Trans_r = camtoworlds[0,:3,3]-camtoworlds[-1,:3,3]
    Trans_n = c2w[0,:3,3]-c2w[-1,:3,3]

    K = np.stack(cam_intrinsics,axis=0)
    poses = c2w[:, :3, :4].transpose([1,2,0])

    cx = K[:,0,2]+0.5
    cy = K[:,1,2]+0.5
    f = K[:,0,0]

    cam_K = np.stack([cx,cy,f],axis=0)

    poses = np.concatenate([poses,cam_K[:,np.newaxis,:]], 1)
    poses = np.concatenate([poses[:, 1:2, :],poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    # Change the image array to img
    print('the number of total images:',len(images))
    path = os.path.join(args.savedir, 'images')
    os.makedirs(path,exist_ok=True)
    # Resize the image to the destermined resolution
    if args.resize:
        width=512
        height=288
    else:
        width=1600
        height=900

    if args.resize:
        for i in range(images.shape[0]):
            img_resized = cv2.resize(images[i], (width, height))
            cv2.imwrite(path + '/{:04d}.png'.format(i),img_resized)
        poses[-1,-1,:]*=height/images.shape[1]
        poses[1,-1,:]*=height/images.shape[1]
        poses[0,-1,:]*=height/images.shape[1]
    else:
        for i in range(len(images)):
            cv2.imwrite(path+'/{:04d}.png'.format(i),images[i])
    save_arr = []
    for i in range(poses.shape[-1]):
        close_depth=1;inf_depth=99.9
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth]),np.array([height,width])], 0))

    
    # Save
    # The shape of poses_bounds:n,19
    # The first 15 nums consist of the poses: [R | T | cam_K]
    # The last fours contain cls and inf depth, the height and width of images.
    save_arr = np.array(save_arr)
    save_path = os.path.join(args.savedir, 'poses_bounds.npy')
    save_json = os.path.join(args.savedir, 'toekn.json')
    np.save(save_path, save_arr)
    with open(save_json, 'w+') as f:
        json.dump(sample_idx_list, f)
