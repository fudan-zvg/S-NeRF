'''
Generate calibs.npy and intrinsic.npy for foreground rendering.
'''

import os
import yaml
import random
import argparse
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

from utils_render import *

def find_category(nums_list, idx):
    category_list = ["vehicle", "person", "object", "bicycle", "motorcycle"]
    _CDF = [0] + [sum(nums_list[:_]) for _ in range(1, len(nums_list)+1)]
    _temp = [idx + 1 - ct for ct in _CDF]
    for i in range(len(_temp)):
        diff = _temp[i]
        if diff <= 0:
            target_i = i - 1
            break
    category = category_list[target_i]
    category_idx = _temp[target_i] - 1
    
    return category, category_idx

def get_no_sample_idx(rgb_dir, valid_HW=(1280, 1920)):
    '''
    Find the invalid HW imgs as invalid idx.
    '''
    rgb_files = sorted(os.listdir(rgb_dir))
    invalid_idx = []
    for idx, rgb_f in enumerate(rgb_files):
        assert idx == int(rgb_f[:-4])
        rgb_path = os.path.join(rgb_dir, rgb_f)
        if np.array(Image.open(rgb_path)).shape[:2] != valid_HW:
            invalid_idx.append(idx)

    return invalid_idx

def random_noise(x, dx):
    return random.uniform(x-dx, x+dx)

def random_interval(interval):
    a, b = interval
    a, b = min(a, b), max(a, b)
    return random.uniform(a, b)

def random_genrate_trajs(n_images, reverse=False):
    # Randomly generate trajectories.
    x_range = [2780, 2750]
    y_range = [3598, 3600]
    z_range = [-46.25, -46]
    
    dx = 3
    
    # n_images = 300    # Special case.
    
    #### Generate. ####
    start_x = random_noise(x_range[0], dx)
    end_x = random_noise(x_range[1], dx)
    
    start_y = random_interval(y_range)
    end_y = random_interval(y_range)
    start_z = random_interval(z_range)
    end_z = random_interval(z_range)
    
    world_coord_start = [start_x, start_y, -48.225]
    world_coord_end = [end_x, end_y, -48.225]
    
    angle = atan2(end_y-start_y, end_x-start_x) / (2*pi) * 360
    
    world_coord_list = [(idx/(n_images-1) * np.array(world_coord_end) + (1 - idx/(n_images-1)) * np.array(world_coord_start)).tolist() for idx in range(n_images)]
    base_angle_list = [angle] * n_images
        
    if reverse:
        world_coord_list = world_coord_list[::-1]
        base_angle_list = [angle-180 for angle in base_angle_list[::-1]]
    
    return world_coord_list, base_angle_list

def random_generate_pos(n_images, vehicles_n, min_dist=20):
    
    def dist(pos1, pos2):
        '''
        pos: [x, y, z].
        '''
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    x_range = [2790, 2750]
    y_range = [3597, 3601]
    # z_range = [-46.25, -46]
    height = -48.225
    angle_range = [0, 360]
    
    world_coord_list_vehicles, base_angle_list_vehicles =  [[] for _ in range(vehicles_n)], [[] for _ in range(vehicles_n)]
    for idx in range(n_images):
        temp = []
        for vehicle_idx in range(vehicles_n):
            while True:
                x, y, z = random_interval(x_range), random_interval(y_range), height
                one_pos = [x, y, z]
                for already_pos in temp:
                    if dist(already_pos, one_pos) < min_dist:   # To avoid the overlap of vehicles.
                        break                                   # Generate a new pos.
                else:
                    break
            temp.append(one_pos)
            world_coord_list_vehicles[vehicle_idx].append(one_pos)
            base_angle_list_vehicles[vehicle_idx].append(random_interval(angle_range))
    
    return world_coord_list_vehicles, base_angle_list_vehicles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_images', type=int, default=-1)
    parser.add_argument('--bg_name', type=str, required=True)
    parser.add_argument('--vehicles_n', type=int, required=True)
    parser.add_argument('--persons_n', type=int, required=True)
    parser.add_argument('--objects_n', type=int, required=True)
    parser.add_argument('--bicycles_n', type=int, required=True)
    parser.add_argument('--motorcycles_n', type=int, required=True)
    parser.add_argument('--render_factor', type=int, default=4)
    parser.add_argument('--wkdir', type=str, default="wkdir_0", required=True)    
    parser.add_argument('--dataset', type=str, default="waymo")
    parser.add_argument('--mesh_render', type=int, default=0)
    parser.add_argument('--demo', type=int, default=0)

    args = parser.parse_args()
    
    root_dir = '.'
    wkdir = args.wkdir
    n_images = args.n_images
    out_dir = join(root_dir, wkdir, 'out_dir_stage0')
    os.makedirs(out_dir, exist_ok=True)
    bg_dir = join(root_dir, wkdir, 'raw_data', 'background', args.bg_name)
    rgb_dir = join(bg_dir, "rgb")
    vehicles_n = args.vehicles_n
    persons_n = args.persons_n
    objects_n = args.objects_n
    bicycles_n = args.bicycles_n
    motorcycles_n = args.motorcycles_n
    nums_list = [vehicles_n, persons_n, objects_n, bicycles_n, motorcycles_n]
    instances_n = sum(nums_list)
    mode = "NeuS"
    pitch_angle = 160
    yaw_angle = 200
    align = False
    use_trajs = False
    render_factor = args.render_factor
    render_poses = np.load(join(bg_dir, "raw_target_poses.npy"))
    intrinsic = np.load(join(bg_dir, "intrinsic.npy"))
    dataset = args.dataset
    valid_HW = (1280, 1920) if dataset == "waymo" else (900, 1600)

    if not args.demo:
        drop_ratio = 1 - (5 / (render_poses.shape[0] + 7))
        points, semantics = get_semantic_points(bg_dir, drop_ratio=drop_ratio, max_depth=60., valid_HW=valid_HW)
        drivable_semantic_idx = 0
        # undrivable_semantic_idx_list = None
        undrivable_semantic_idx_list = [2, 3, 13, 14, 15]
        obstacle_semantic_idx_list = [13, 14, 15]
        no_sample_idx = get_no_sample_idx(rgb_dir, valid_HW=valid_HW)     # Get the side camera, which has a resolution of (1920, 886)
        world_coord_list_vehicles, base_angle_list_vehicles, new_render_poses, new_n_images, bev_results, valid_idx, invalid_idx = \
            generate_pos_from_render_poses(render_poses,
                                           intrinsic,
                                           points,
                                           semantics,
                                           drivable_semantic_idx,
                                           undrivable_semantic_idx_list,
                                           obstacle_semantic_idx_list,
                                           instances_n=instances_n,
                                           depth_range=(7,40),
                                           reject_r=3,
                                           min_dist=5,
                                           no_sample_idx=no_sample_idx)
        # print("New n_images: %d" % new_n_images)
        assert new_n_images >= n_images
        n_images = n_images if n_images != -1 else new_n_images
        # Save the new render poses.
        np.save(join(bg_dir, "target_poses.npy"), new_render_poses)
        cv2.imwrite(join(bg_dir, "bev_map.png"), bev_results["bev_map"])
        cv2.imwrite(join(bg_dir, "bev_map_refined.png"), bev_results["bev_map_refined"])
        np.save(join(bg_dir, "valid_idx.npy"), np.array(valid_idx))
        np.save(join(bg_dir, "invalid_idx.npy"), np.array(invalid_idx))
        np.save(join(bg_dir, "bev_results.npy"), bev_results)
        np.save(join(bg_dir, "points.npy"), points)
        np.save(join(bg_dir, "semantics.npy"), semantics)

    else:
        drop_ratio = 1 - (5 / (render_poses.shape[0] + 7))
        points, semantics = get_semantic_points(bg_dir, drop_ratio=drop_ratio, max_depth=60., valid_HW=valid_HW)
        drivable_semantic_idx = 0
        # undrivable_semantic_idx_list = None
        undrivable_semantic_idx_list = [2, 3, 13, 14, 15]
        obstacle_semantic_idx_list = [13, 14, 15]
        no_sample_idx = get_no_sample_idx(rgb_dir, valid_HW=valid_HW)     # Get the side camera, which has a resolution of (1920, 886)
        world_coord_list_vehicles, base_angle_list_vehicles, new_render_poses, new_n_images, bev_results, valid_idx, invalid_idx = \
            demo_poses_generator(render_poses,
                                           intrinsic,
                                           points,
                                           semantics,
                                           drivable_semantic_idx,
                                           undrivable_semantic_idx_list,
                                           obstacle_semantic_idx_list,
                                           instances_n=instances_n,
                                           depth_range=(7,40),
                                           reject_r=3,
                                           min_dist=5,
                                           no_sample_idx=no_sample_idx)
        # print("New n_images: %d" % new_n_images)
        assert new_n_images >= n_images
        n_images = n_images if n_images != -1 else new_n_images
        # Save the new render poses.
        np.save(join(bg_dir, "target_poses.npy"), new_render_poses)
        cv2.imwrite(join(bg_dir, "bev_map.png"), bev_results["bev_map"])
        cv2.imwrite(join(bg_dir, "bev_map_refined.png"), bev_results["bev_map_refined"])
        np.save(join(bg_dir, "valid_idx.npy"), np.array(valid_idx))
        np.save(join(bg_dir, "invalid_idx.npy"), np.array(invalid_idx))
        np.save(join(bg_dir, "bev_results.npy"), bev_results)
        np.save(join(bg_dir, "points.npy"), points)
        np.save(join(bg_dir, "semantics.npy"), semantics)


    for instance_idx in range(instances_n):
        world_coord_list, base_angle_list = world_coord_list_vehicles[instance_idx], base_angle_list_vehicles[instance_idx]
        calib_list = []
        P_list = []
        c2w_list = []
        plane_coord_list = []
        ct = 0
        for idx in range(n_images):
            
            #### Calculate the moving vehicle ####
            world_coord = world_coord_list[idx]
            base_angle = base_angle_list[idx]
            calib, P, c2w = get_camcalib(bg_dir,
                                         idx,
                                         world_coord,
                                         base_angle,
                                         mode="NeuS")
            
            calib_list.append(calib)
            P_list.append(P)   
            c2w_list.append(c2w)    
            
            print("Finish: %d." % ct)   
            ct += 1
            
        calibs = np.stack(calib_list, axis=0)
        Ps = np.stack(P_list, axis=0)
        c2ws = np.stack(c2w_list, axis=0)
        # print(calibs.shape)
        # print(Ps.shape)
        
        # if find_category(nums_list, instance_idx)[0] == "vehicle":
        #     np.save(join(out_dir, 'calibs_%d.npy' % instance_idx), calibs)
        #     np.save(join(out_dir, "intrinsic_%d.npy" % instance_idx), Ps)   
              
        #############################################
        # Note that the intrinsics have been scaled! 
        #############################################
        
        x = calibs[:, 0, 3]
        z = calibs[:, 2, 3]
        
        
        # Write the meta data.
        # if find_category(nums_list, instance_idx) == "vehicle":
        #     category = "vehicle"
        #     fg_dir = join('raw_data', 'foreground', 'vehicle_%d' % get_category_idx(nums_list, instance_idx))
        # elif find_category(nums_list, instance_idx) == "person":
        #     category = "person"
        #     fg_dir = join('raw_data', 'foreground', 'person_%d' % get_category_idx(nums_list, instance_idx))
        # elif find_category(nums_list, instance_idx) == "object":
        #     category = "object"
        #     fg_dir = join('raw_data', 'foreground', 'object_%d' % get_category_idx(nums_list, instance_idx))        
        category, category_idx = find_category(nums_list, instance_idx)
        fg_dir = join(root_dir, wkdir, 'raw_data', 'foreground', '%s_%d' % (category, category_idx))
        
        np.save(join(out_dir, 'calibs_%s_%d.npy' % (category, category_idx)), calibs)
        np.save(join(out_dir, "intrinsic_%s_%d.npy" % (category, category_idx)), Ps)         
        
        os.makedirs(fg_dir, exist_ok=True)
        mesh_dir = join(fg_dir, 'mesh.ply')
        data =  {
            "bg_name": args.bg_name,
            "category": category, 
            "fg_dir" : fg_dir,
            "world_coord_list" : world_coord_list,
            "base_angle_list": base_angle_list,
            "mesh_dir": mesh_dir,
            "mesh_bias": [0, 0, 0],
            "pitch_angle": pitch_angle,
            "yaw_angle": yaw_angle,
            "mode": mode,
            "align": align,
            "mesh_render": args.mesh_render,
        }
        
        fw = open(join(fg_dir, './meta_data.yaml'), 'w')
        yaml.dump(data, fw)
        fw.close()
        
    # Last save intrinsics as for scaled intrinsic.
    
    np.save(join(out_dir, "intrinsic.npy"), Ps) 
    np.save(join(out_dir, "c2w.npy"), c2ws)
    
    #############################################
    # Note that the intrinsics have been scaled! 
    #############################################    