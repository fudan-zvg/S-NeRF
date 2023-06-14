
import numpy as np
import cv2
import os
from PIL import Image
import argparse
from pose import get_ego_pose, convert_pose

def generate_poses(args):
    K = np.load(f'{args.scene_name}/intrinsic.npy').astype(np.float32)
    c2w = np.load(f'{args.scene_name}/c2w.npy').astype(np.float32)
    
    # We drop the first image, because the flow of the first image is not able to be calculated
    c2w = c2w[:,1:1+img_num,...]
    K = K[:,1:1+img_num,...]
    c2w = c2w.reshape(-1,4,4)
    K = K.reshape(-1,3,3)

    c2w = np.linalg.inv(c2w[0]) @ c2w
    hwf = np.stack([K[:,0,2],K[:,1,2],(K[:,0,0]+K[:,1,1])/2],axis=1)

    poses = np.concatenate([c2w[:,:3,:4],hwf[:,:,None]],axis=-1)
    poses = np.concatenate([poses[:, :,1:2] ,poses[:, :,0:1], -poses[:, :,2:3], poses[:,:, 3:4], poses[:, :,4:5]], -1)
    return poses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default = '')    # the file pathname  of data in kitti format
    parser.add_argument('--scene_name', type=str, default = '') # the file pathname  of data in Snerf format
    parser.add_argument('--depthdir',type= str, default= '')    # the file pathname  of depth data 

    parser.add_argument('--img_num', type=int, default=50)      # images num for every cam
    parser.add_argument('--height', type=int, default=1280)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--near', type=int, default=1)
    parser.add_argument('--far', type=int, default=100)

    #*------ Load configuration ------*#
    args = parser.parse_args()
    os.makedirs(args.scene_name, exist_ok=True)
    imgdir = args.datadir
    depthdir = args.depthdir
    scene_name = args.scene_name

    images = [];depths = []
    img_num = args.img_num
    height = args.height;width = args.width
    close_depth = args.near;inf_depth = args.far

    #* 1. pose processing
    print('pose processing')
    try:
        ego_poses = get_ego_pose(args)
        K, cam2lidar = convert_pose(args)
        c2w = ego_poses @ cam2lidar
        np.save(os.path.join(args.scene_name,'c2w.npy'), c2w)
        np.save(os.path.join(args.scene_name,'intrinsic.npy'), K)
        poses = generate_poses(args)
        save_poses=[]
        for i in range(poses.shape[0]):
            save_poses.append(
                np.concatenate([poses[i,...].ravel(), 
                np.array([close_depth, inf_depth]),
                np.array([height, width])], 0))
        np.save(os.path.join(scene_name,'poses_bounds.npy'),save_poses)
        print('Saved poses')
    except FileNotFoundError:
        print('Poses Not Found')

    #* 2. image processing
    print('images processing')
    try:
        for i in ['image_0', 'image_1', 'image_2','image_3','image_4']:
            k = 0
            for j in sorted(os.listdir(os.path.join(imgdir,i)), key=lambda x:int(x.split('.')[0])):
                padding=np.zeros((height,width,3))
                if i in ['image_0', 'image_1', 'image_2']:
                    images.append(cv2.imread(os.path.join(imgdir,i,j)))
                else:
                    padding[:886,...]=cv2.imread(os.path.join(imgdir,i,j))
                    images.append(padding)
                k += 1
                if k > img_num:
                    break
        images=np.stack(images,axis=0)
        save_imgs=images.reshape(5,-1,height,width,3)
        save_imgs=save_imgs[:,1:,...]
        save_imgs=save_imgs.reshape(-1,height,width,3)
        os.makedirs(os.path.join(scene_name,'images'),exist_ok=True)
        for i in range(save_imgs.shape[0]):
            cv2.imwrite(os.path.join(scene_name,'images','{:04d}.png'.format(i)),save_imgs[i])
        print('Saved images')
    except FileNotFoundError:
        print('Images Not found')

    #* 3. depth processing
    print('depth processing')
    try:
        for i in ['image_0', 'image_1', 'image_2','image_3','image_4']:
            k = 0
            for j in sorted(os.listdir(os.path.join(depthdir,i)), key=lambda x:int(x.split('.')[0])):
                padding=np.zeros((height,width))
                if i in ['image_0', 'image_1', 'image_2']:
                    depths.append(cv2.imread(os.path.join(depthdir,i,j),-1))
                else:
                    padding[:886,...]=cv2.imread(os.path.join(depthdir,i,j),-1)
                    depths.append(padding)
                k += 1
                if k > img_num-1:
                    break

        depths=np.stack(depths,axis=0)
        os.makedirs(os.path.join(scene_name,'depth'),exist_ok=True)

        for j in range(len(depths)):
            Image.fromarray((depths[j]).astype(np.uint16)).save(os.path.join(scene_name,'depth','{:04d}.png'.format(j)))
        print('Saved depth')
    except FileNotFoundError:
        print('Depth Not Found')

        
    
