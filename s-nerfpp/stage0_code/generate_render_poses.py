'''
Generate render_poses.npy for background rendering.
Generate randomly.
'''

import argparse
import numpy as np
from os.path import join
import random
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from math import atan2, pi

from utils_render import *

def random_noise(x, dx):
    return random.uniform(x-dx, x+dx)

def random_interval(interval):
    a, b = interval
    a, b = min(a, b), max(a, b)
    return random.uniform(a, b)

def random_generate_trajs(n_images):
    
    #### Args. ####
    x_range = [2830, 2800]
    y_range = [3598, 3600]
    z_range = [-46.5, -46]
    
    dx = 3
    
    #### Generate. ####
    start_x = random_noise(x_range[0], dx)
    end_x = random_noise(x_range[1], dx)
    
    start_y = random_interval(y_range)
    end_y = random_interval(y_range)
    start_z = random_interval(z_range)
    end_z = random_interval(z_range)
    
    # x_degree, y_degree = random_interval(x_degree_range), random_interval(y_degree_range)
    
    # y_rot = R.from_euler('y', y_degree, degrees=True).as_matrix()    
    # x_rot = R.from_euler('x', x_degree, degrees=True).as_matrix()
    
    world_coord_start = [start_x, start_y, start_z]
    world_coord_end = [end_x, end_y, end_z]
    
    angle = atan2(end_x-start_x, end_y-start_y)  # Compute the angle between y-axis.
    rot = R.from_euler('xz', (-pi/2, -angle)).as_matrix()
    
    trans_list = [(idx/299 * np.array(world_coord_end) + (1 - idx/299) * np.array(world_coord_start)).tolist() for idx in range(300)]
    
    render_poses = np.expand_dims(np.eye(4), axis=0).repeat(n_images, axis=0)
    render_poses[:, :3, :3] = rot @ render_poses[:, :3, :3]
    render_poses[:, :3, 3] = trans_list
    
    return render_poses    

def random_generate_pos(n_images):
    
    #### Args. ####
    x_range = [2815, 2805]
    y_range = [3597, 3601]
    z_range = [-46.5, -46]
    
    base_view_angle = -1.50423
    d_theta = 0.25
    
    render_poses = np.expand_dims(np.eye(4), axis=0).repeat(n_images, axis=0)
    for idx in range(n_images):
        
        view_angle = random_noise(base_view_angle, d_theta)
        x, y, z = random_interval(x_range), random_interval(y_range), random_interval(z_range)
        
        rot = R.from_euler('xz', (-pi/2, -view_angle)).as_matrix()
        render_poses[idx, :3, :3] = rot @ render_poses[idx, :3, :3]
        render_poses[idx, :3, 3] = [x, y, z]
    
    return render_poses

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", default=1000, type=int)
    parser.add_argument("--use_trajs", default=False, action="store_true")
    
    args = parser.parse_args()
    
    # Randomly choose camera trace.
    out_dir = "./out_dir_stage0"
    
    #### Args. ####
    
    n_images = args.n_images
    use_trajs = args.use_trajs
    
    #### Generate. ####

    if use_trajs:
        render_poses= random_generate_trajs(n_images)
    else:
        render_poses= random_generate_pos(n_images)
    
    np.save(join(out_dir, "target_poses.npy"), render_poses)