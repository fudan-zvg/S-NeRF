'''
Generate depth map, bound, fuse results, mask data.
'''

import os
import json
import glob
import yaml
import numpy as np
import torch
import argparse
import multiprocessing
from os.path import join
from math import atan2, pi, sin, cos
from copy import deepcopy

from utils_render import *

# Create the out dirs. 

def create_dirs(n_vehicles):
    def create_dir(dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
            print("Create dir: %s" % dir)
        else:
            f_list = os.listdir(dir)
            for f in f_list:
                os.remove(join(dir, f))
            print("Remove all files in %s" % dir)
    
    create_dir(join(out_dir, "bound"))
    create_dir(join(out_dir, "depth"))
    create_dir(join(out_dir, "semantic"))
    create_dir(join(out_dir, "bbox"))
    create_dir(join(out_dir, "mask"))
    create_dir(join(out_dir, "fuse"))
    create_dir(join(out_dir, "occluded_mask"))

    for vehicle_idx in range(n_vehicles):
        create_dir(join(out_dir, "bound_%d" % vehicle_idx))
        create_dir(join(out_dir, "mask_%d" % vehicle_idx))     
        

def read_conf(fg_dir):
    conf_file = join(fg_dir, "meta_data.yaml")
    with open(conf_file, 'r', encoding='utf-8') as f:
        conf = f.read()
        yamlconfig = yaml.load(conf, Loader=yaml.FullLoader)
    return yamlconfig
        
        
def main(out_dir,
         bg_dir,
         idx_list,
         fg_data_dict,
         render_factor,
         valid_idx_list=None,
         args=None):
    
    #### Process. ####
    n_instances = len(fg_data_dict)  
    mesh_list =  [trimesh.load(data_dict['mesh_dir']) for data_dict in fg_data_dict]
    
    # Process.
    for idx in idx_list:
                
        total_fuse_im = None
        total_bound_im = None
        total_mask_im = None
        total_depth_mat = None
        total_semantic_mat = None
        total_occluded_mask_im = None
        total_bbox_result_list = []
        
        # Find out the occlusion.
        mesh_list_copied = deepcopy(mesh_list)
        data_dict = fg_data_dict[0]
        mode, align = data_dict.get('mode', "NeuS"), data_dict.get('align', False)
        instance_idx_list = occlution_order(idx, fg_data_dict, list(range(n_instances)), bg_dir, mesh_list_copied, render_factor, mode, align,
                                            meta_data=data_dict)
        
        for instance_idx in instance_idx_list:
            
            # Read the foreground data.
            data_dict = fg_data_dict[instance_idx]
            fg_dir = data_dict['fg_dir']
            world_coord_list = data_dict['world_coord_list']
            base_angle_list = data_dict['base_angle_list']
            mesh_dir = data_dict['mesh_dir']    
            category = data_dict.get('category', "vehicle")
            # mode = data_dict.get('mode', "NeuS")   
            # align = data_dict.get('align', False)
            
            #### Calculate the moving vehicle ####
            world_coord = world_coord_list[idx]
            base_angle = base_angle_list[idx]
            
            ##########################################################################
            plane_coord, im = find_position_rgb(bg_dir,
                                                idx,
                                                world_coord,
                                                valid_idx_list)
            
            calib, P, c2w = get_camcalib(bg_dir,
                                         idx,
                                         world_coord,
                                         base_angle,
                                         mode=mode)
            
            depth_mat, semantic_mat = get_depth(idx, bg_dir, valid_idx_list), get_semantic(idx, bg_dir, valid_idx_list)
            

            # inverse_flag = False if vehicle_idx != 1 else True
            if mode == "SNeRF":
                inverse_flag = False
                sc = cal_sc(idx, mesh_dir, fg_dir, bg_dir, world_coord, base_angle, inverse_flag)          
                # If not want to use calculated sc, let sc = None.
                instance, mask_im = decode_image(fg_dir, idx, P, plane_coord, render_factor, inverse_flag, sc, mode=mode,
                                                 category=category, meta_data=data_dict)
            else:
                inverse_flag = False
                # sc = None         # Load sc.
                sc = 1              # Now we don't need to handle scale here.
                if align:
                    sc = cal_sc(idx, mesh_dir, fg_dir, bg_dir, world_coord, base_angle, inverse_flag, mode=mode)
                instance, mask_im = decode_image(fg_dir, idx, P, plane_coord, render_factor, inverse_flag, sc, mode=mode, align=align, category=category,
                                                 meta_data=data_dict)
            
            _, j_list, _ = np.where(np.array(mask_im) > 0)
            if j_list.size == 0:
                r = 1
            else:
                mask_length = np.max(j_list) - np.min(j_list)
                r = int((mask_length / 80) ** .82)
                if category in ['motorcycle', 'bicycle']:
                    r = 3       
            
            total_fuse_im, depth_mat, semantic_mat, bbox_result, occlution_per, occluded_mask_im = fuse(idx,
                                                                                                        bg_dir,
                                                                                                        world_coord,
                                                                                                        base_angle,
                                                                                                        depth_mat if total_depth_mat is None else total_depth_mat,
                                                                                                        semantic_mat if total_semantic_mat is None else total_semantic_mat,
                                                                                                        im if not total_fuse_im else total_fuse_im,
                                                                                                        instance,
                                                                                                        mask_im,
                                                                                                        mesh=deepcopy(mesh_list)[instance_idx],
                                                                                                        category=category)               
            
            # First save the individual data.
            bound_im = get_bound_im(mask_im, r)
            mask_im = set_diff(mask_im, bound_im)
            bound_im.save(join(out_dir, "bound_%d" % instance_idx, '%05d.png' % idx))
            mask_im.save(join(out_dir, "mask_%d" % instance_idx, '%05d.png' % idx))
            
            # Then fuse the data.
            total_bound_im = bound_im if not total_bound_im else fuse_bound(total_mask_im, total_bound_im, bound_im, mask_im)
            total_mask_im = mask_im if not total_mask_im else Image.fromarray((np.array(mask_im).astype(bool) | np.array(total_mask_im).astype(bool)).astype(np.uint8) * 255)
            if category == "vehicle":          
                # This is for light handling.
                # We only do this for vehicles.
                total_occluded_mask_im = occluded_mask_im if not total_occluded_mask_im\
                    else Image.fromarray((np.array(occluded_mask_im).astype(bool) | np.array(total_occluded_mask_im).astype(bool)).astype(np.uint8) * 255)
            total_depth_mat = depth_mat.copy()
            total_semantic_mat = semantic_mat.copy()
            total_bbox_result_list.append(bbox_result)
        
        # Get the final bound and add to the fuse_im.
        # total_bound_im = get_bound_im(total_mask_im, r)
        #### ablation study save for no postprocess image
        # import time
        # os.makedirs('./annotation_no_process', exist_ok=True)
        # result_dir = join('./annotation_no_process', args.bg_name, )
        # os.makedirs(result_dir, exist_ok=True)
        # result_dir = join(result_dir, args.wkdir)
        # os.makedirs(result_dir, exist_ok=True)
        # total_fuse_im.save(join(result_dir, '%05d.png' % idx))

        total_fuse_im = fuse_bound_and_im(total_fuse_im, total_bound_im)
        
        # Save the data.
        total_mask_im.save(join(out_dir, "mask", '%05d.png' % idx))
        if total_occluded_mask_im is None:
            total_occluded_mask_im = Image.fromarray(np.zeros_like(np.array(total_mask_im)))
        total_occluded_mask_im.save(join(out_dir, "occluded_mask", "%05d.png" % idx))
        total_bound_im.save(join(out_dir, "bound", '%05d.png' % idx))
        total_fuse_im.save(join(out_dir, "fuse", '%05d.png' % idx))
        save_bbox_result_for_one_frame(join(out_dir, "bbox", "%05d.txt" % idx), total_bbox_result_list)
        Image.fromarray((total_depth_mat * 256).astype(np.uint16)).save(join(out_dir, 'depth', '%05d.png' % idx))
        Image.fromarray(total_semantic_mat).save(join(out_dir, 'semantic', '%05d.png' % idx))
        # np.save(join(out_dir, "depth", "%05d.png" % idx), total_depth_mat)
        # np.save(join(out_dir, "semantic", "%05d.png" % idx), total_semantic_mat)
        
            
        print("Finish: %d." % idx)     
        


if __name__ == '__main__':
    
    root_dir = '.'
    #### Args. ####
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--n_images', type=int, default=1000)
    parser.add_argument('--bg_name', type=str, required=True)
    parser.add_argument('--vehicles_n', type=int, required=True)
    parser.add_argument('--persons_n', type=int, required=True)
    parser.add_argument('--objects_n', type=int, required=True)
    parser.add_argument('--bicycles_n', type=int, required=True)
    parser.add_argument('--motorcycles_n', type=int, required=True)
    parser.add_argument('--render_factor', type=int, default=4)
    parser.add_argument('--n_threads', type=int, default=10)
    parser.add_argument('--n_images', type=int, default=-1)
    parser.add_argument('--wkdir', type=str, default="wkdir_0", required=True)      
    
    args = parser.parse_args()
    
    # Background and base.
    wkdir = args.wkdir
    out_dir = join(root_dir, wkdir, 'out_dir_stage1')
    os.makedirs(out_dir, exist_ok=True)
    bg_dir = join(root_dir, wkdir, 'raw_data', 'background', args.bg_name)
    render_factor = args.render_factor
    n_images_max = np.load(join(bg_dir, 'target_poses.npy')).shape[0]
    n_images = args.n_images if args.n_images != -1 and args.n_images <= n_images_max\
        else n_images_max   # FIXME!
    
    # Insert one tensor to gpu.
    _ = torch.tensor([0])
    _.cuda()
    
    # For foreground data.
    # fg_dir_list = [
    #     join(root_dir, 'raw_data', 'foreground', '7'),
    #     join(root_dir, 'raw_data', 'foreground', 'nerf_data_moving2_7')
    # ]    
    vehicles_n = args.vehicles_n
    persons_n = args.persons_n
    objects_n = args.objects_n
    bicycles_n = args.bicycles_n
    motorcycles_n = args.motorcycles_n
    instances_n = vehicles_n + persons_n + objects_n + bicycles_n + motorcycles_n
    fg_root_dir = join(root_dir, wkdir, 'raw_data', 'foreground')
    # fg_dir_list = [join(fg_root_dir, dir_name) for dir_name in sorted(os.listdir(fg_root_dir))\
    #     if dir_name[:8] == "vehicle_" or dir_name[:7] == "person_"][:instances_n]
    # vehicle_dir_list = [join(fg_root_dir, dir_name) for dir_name in sorted(os.listdir(fg_root_dir))\
    #     if dir_name[:8] == 'vehicle_'][:vehicles_n]
    # person_dir_list = [join(fg_root_dir, dir_name) for dir_name in sorted(os.listdir(fg_root_dir))\
    #     if dir_name[:7] == "person_"][:persons_n]
    fg_dir_list = []
    category_list = ["vehicle", "person", "object", "bicycle", "motorcycle"]
    nums_list = [vehicles_n, persons_n, objects_n, bicycles_n, motorcycles_n]
    for category, num in zip(category_list, nums_list):
        fg_dir_list = fg_dir_list + [join(fg_root_dir, dir_name) for dir_name in sorted(os.listdir(fg_root_dir))\
         if '%s_' % category in dir_name][:num]
    fg_data_dict = [
        read_conf(fg_dir) for fg_dir in fg_dir_list
    ]
    
    ################
    
    # Make dirs.
    create_dirs(instances_n)
    
    # Copy the camera data to the out_dir.
    path = join(bg_dir, "target_poses.npy")
    out_path = join(out_dir, "target_poses.npy")
    os.system("cp %s %s" % (path, out_path))
    path = join(bg_dir, "intrinsic.npy")
    out_path = join(out_dir, "intrinsic.npy")
    os.system("cp %s %s" % (path, out_path))    
    
    # Multiprocess.
    n_threads = args.n_threads
    total_idx_list = list(range(n_images))
    valid_idx_list = np.load(join(bg_dir, "valid_idx.npy")).tolist()
    threads_list = []
    
    try:
        multiprocessing.set_start_method('spawn')
    except:
        pass
    ################################
    for id in range(n_threads):
        step = n_images//n_threads
        partial_list = total_idx_list[id * step : (id+1) * step]
        if id == (n_threads-1): partial_list = total_idx_list[id * step :]
        one_process = multiprocessing.Process(target=main, kwargs={
            "out_dir": out_dir,
            "bg_dir": bg_dir,
            "idx_list": partial_list,
            "fg_data_dict": fg_data_dict,
            "render_factor": render_factor,
            "valid_idx_list": valid_idx_list,
            "args": args
        })
        one_process.start()
        threads_list.append(one_process)
        print('Process %d starts.' % id)

    for thread in threads_list:
        thread.join()
    #################################

    # if True:   # For debugging.
    #     queue = multiprocessing.Queue()
    #     main(out_dir,
    #          bg_dir,
    #          total_idx_list,
    #          fg_data_dict,
    #          render_factor,
    #          valid_idx_list)