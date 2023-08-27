from imageio import save
import numpy as np
import os
import cv2
import json

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from collections import defaultdict
import argparse

SENSORS = [
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-mini', 'v1,0-trainval'])
    parser.add_argument('--datadir', type=str, default = './data' )
    parser.add_argument('--skip', type=int, default=20, help = 'Skip the first N frames') # Caution, this may vary from scene to scene
    parser.add_argument('--total_num', type=int, default = 40, help = 'The frames needed')
    parser.add_argument('--camera_index', type=list, default = [0,1,2,3,4,5], help = 'Cameras chosen')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--savedir', type=str, default='./data/scenes')
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=900)
    args = parser.parse_args()

    # Data Initialization
    dataroot = os.path.join(args.datadir, 'nuScenes', args.version.split('-')[-1])
    nusc = NuScenes(version=args.version, dataroot=dataroot, verbose=True)
    sensor = [SENSORS[i] for i in args.camera_index]
    scene_dict_path = os.path.join(args.datadir, 'scene_dict.json')
    with open(scene_dict_path, 'r') as f:
        scene_dict = json.load(f)
    
    for scene_name, scene_token in scene_dict.items():
        images, ego2global_rts, cam2ego_rts, cam_intrinsics = [], [], [], []
        print(f"Start processing scene {scene_name} | token: {scene_token}")
        scene = nusc.get('scene', scene_token)
        assert(scene['name'] == scene_name)
        temp_sample = nusc.get('sample', scene['first_sample_token'])
        for i in range(args.skip):
            temp_sample = nusc.get('sample', temp_sample['next'])
        IDX = 0
        sample_idx_list = {}
        channel_tokens = defaultdict(list)
        for s in sensor:
            temp_data = nusc.get('sample_data', temp_sample['data'][s])
            for i in range(args.total_num):      
                data_path , _ , cam_intrinsic = nusc.get_sample_data(temp_data['token'])
                if not os.path.exists(data_path):
                    import pdb; pdb.set_trace()
                
                channel_tokens[s].append(temp_data['token'])
                if(temp_data['is_key_frame']):
                    sample_idx_list[IDX] = temp_data['token']
                IDX += 1

                cam_intrinsics.append(cam_intrinsic.astype(np.float32))
                #image
                fname = data_path
                img = cv2.imread(fname)
                images.append(img)
                #ego2global
                temp_ego2global = nusc.get('ego_pose',temp_data['ego_pose_token'])
                ego2global_r = Quaternion(temp_ego2global['rotation']).rotation_matrix
                ego2global_t = np.array(temp_ego2global['translation'])
                ego2global_rt = np.eye(4)
                ## correct the ego2global pose
                ego2global_rt[:3,:3] = ego2global_r
                ego2global_rt[:3,3] = ego2global_t
                
                ego2global_rts.append(ego2global_rt.astype(np.float32))
                temp_cam2ego = nusc.get('calibrated_sensor',temp_data['calibrated_sensor_token'])
                #cam2ego
                cam2ego_r=Quaternion(temp_cam2ego['rotation']).rotation_matrix
                cam2ego_t=np.array(temp_cam2ego['translation'])
                cam2ego_rt = np.eye(4)
                cam2ego_rt[:3, :3] = cam2ego_r
                cam2ego_rt[:3, 3] = cam2ego_t
                cam2ego_rts.append(cam2ego_rt.astype(np.float32))
                
                if temp_data['next'] == '':
                    print(f"Sensor {s} Done, Total {len(images)} images")
                    break
                temp_data = nusc.get('sample_data', temp_data['next'])

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
        
        #* Save the data
        save_dir = os.path.join(args.savedir, scene_name)
        # Change the image array to img
        print('the number of total images:', len(images))
        path = os.path.join(save_dir, 'images')
        os.makedirs(path,exist_ok=True)
        # Resize the image to the destermined resolution
        
        width = args.width
        height = args.height
        
        for i in range(len(images)):
            cv2.imwrite(path+'/{:04d}.png'.format(i),images[i])

        save_arr = []
        for i in range(poses.shape[-1]):
            close_depth=1; inf_depth=999.9
            save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth]),np.array([height,width])], 0))

        # Save
        # The shape of poses_bounds:n,19
        # The first 15 nums consist of the poses: [R | T | cam_K]
        # The last fours contain cls and inf depth, the height and width of images.
        save_arr = np.array(save_arr)
        save_path = os.path.join(save_dir, 'poses_bounds.npy')
        save_json = os.path.join(save_dir, 'token.json')
        save_tokens = os.path.join(save_dir, 'channel_tokens.json')
        np.save(save_path, save_arr)
        
        with open(save_json, 'w+') as f:
            json.dump(sample_idx_list, f)

        with open(save_tokens, 'w+') as f:
            json.dump(channel_tokens, f)

