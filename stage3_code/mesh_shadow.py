import os
import time
import yaml
import argparse
import numpy as np
from os.path import join
from utils import *
from scipy.spatial.transform import Rotation as R
import multiprocessing
from PIL import ImageDraw


def read_conf(fg_dir):
    conf_file = join(fg_dir, "meta_data.yaml")
    with open(conf_file, 'r', encoding='utf-8') as f:
        conf = f.read()
        yamlconfig = yaml.load(conf, Loader=yaml.FullLoader)
    return yamlconfig


def generate_one_shadow(mesh_points,
                        center,
                        idx_list,
                        instance_idx,
                        dir,
                        im_dir,
                        out_dir,
                        world_coord_list,
                        theta_list,
                        mesh_bias,
                        pitch_angle,
                        yaw_angle,
                        ground_height=None,
                        interpolate_r=20,
                        interpolate_iter=3,
                        refine_r=1,
                        refine_iter=1,
                        light_scale=.55,
                        blur=21,
                        mask_erode=1,
                        check=False,
                        align=True
                        ):

    default_mesh_sc = 1
    
    for idx in idx_list:

        # Read the data.
        im = read_im(im_dir, idx)
        # mask_mat, _ = read_mask_depth_mat(dir, idx, vehicle_idx, mask_erode)
        mask_mat = read_mask_mat(dir, idx, instance_idx, mask_erode)
        total_mask_mat = read_total_mask_mat(dir, idx, mask_erode)
        c2w = read_calib(dir, idx)
        P = read_P(dir, idx)   

        bbox_center = world_coord_list[idx]
        theta = theta_list[idx]

        # 1. Put the mesh to world's coords.
        points_3D, points_center = process_mesh(mesh_points, center, bbox_center, theta, mesh_bias)
        # points_3D = process_mesh(vertix_points, bbox_center, theta, mesh_bias)

        # 1.5 Align the mesh and the bbox_center.
        if not check and align:
            points_3D, points_center, mesh_sc = align_mesh(points_3D, points_center, bbox_center, mask_mat, c2w, P, default_mesh_sc)
            default_mesh_sc = mesh_sc

        # 2. Project the 3D points to the ground.
        if not check:
            points_3D = project_to_ground(points_3D, pitch_angle, yaw_angle, ground_height)

        # 3. Project the 3D points to 2D.
        points_2D = project(points_3D, c2w, P)

        # 4. Generate the 2D shadow mask.
        shadow_mask_mat = points_to_mask(points_2D, mask_mat)

        # 5. Refine the shadow mask.
        shadow_mask_mat = interpolate(shadow_mask_mat, interpolate_r, interpolate_iter)
        
        # 5.5 Check the scale of vehicle.
        if check:
            im = shadow_fuse(im, shadow_mask_mat, 1, blur=None)
            im.save(join(out_dir, 'image', '%05d.png' % idx))
            print("finish: %d" % idx)
            continue

        def blur_shadow():
            shadow_proj_mask = shadow_mask_mat>0
            if shadow_proj_mask.max():
                h, w, _ = shadow_mask_mat.shape
                x, y = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
                w_size = x[shadow_proj_mask.max(-1)].max() - x[shadow_proj_mask.max(-1)].min()
                h_size = y[shadow_proj_mask.max(-1)].max() - y[shadow_proj_mask.max(-1)].min()
                d = 5
                h_size, w_size = max(1, int(h_size // d)), max(1, int(w_size // d))
                blured_shadow = cv2.blur(shadow_proj_mask + 0., ksize=(w_size, h_size))
                print(shadow_proj_mask.sum(), " ----> ", (blured_shadow>0).sum())
                shadow_mask_mat_ = check_the_occlusion(total_mask_mat, blured_shadow)
                masked_blured_shadow = blured_shadow * shadow_mask_mat_ / 255
                weighted_shadow = light_scale * masked_blured_shadow
                im_np = np.array(im)
                im_fuse = im_np * (1 - weighted_shadow)
                return Image.fromarray(np.uint8(im_fuse))
            else:
                return im


        im = blur_shadow()
        im.save(join(out_dir, 'image', '%05d.png' % idx))

        print("finish: %d" % idx)

    

if __name__ == "__main__":     
    
    # Args.
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--n_images', type=int, default=1000)
    parser.add_argument('--bg_name', type=str, required=True)
    parser.add_argument('--vehicles_n', type=int, required=True)
    parser.add_argument('--persons_n', type=int, required=True)
    parser.add_argument('--objects_n', type=int, required=True)
    parser.add_argument('--bicycles_n', type=int, required=True)
    parser.add_argument('--motorcycles_n', type=int, required=True)
    parser.add_argument('--n_threads', type=int, default=10)
    parser.add_argument('--wkdir', type=str, default="wkdir_0", required=True)      
    
    args = parser.parse_args()   
    wkdir = args.wkdir
    dir = "./%s/out_dir_stage1" % wkdir
    im_dir = "./%s/out_dir_stage3" % wkdir
    out_dir = "./%s/out_dir_stage3" % wkdir
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(join(wkdir, 'out_dir_stage3', 'image')):
        os.system("rm -r ./%s/out_dir_stage3/image" % wkdir)
        print("Remove ./%s/out_dir_stage3/image, copying..." % wkdir)
    os.system("cp -r ./%s/out_dir_stage2/image ./%s/out_dir_stage3/image" % (wkdir, wkdir))
    print("Finish copying.")       
        


    root_dir = '.'
    n_images = len(os.listdir(join(im_dir, 'image')))
    idx_list = range(n_images)    
        

    check = False
    align = False
    vehicles_n = args.vehicles_n
    persons_n = args.persons_n
    objects_n = args.objects_n
    bicycles_n = args.bicycles_n
    motorcycles_n = args.motorcycles_n
    instances_n = vehicles_n + objects_n + bicycles_n + motorcycles_n + persons_n   # FIXME: We do not care persons!
    fg_root_dir = join(root_dir, wkdir, 'raw_data', 'foreground')
    # vehicle_dir_list = [join(fg_root_dir, dir_name) for dir_name in sorted(os.listdir(fg_root_dir))\
    #     if dir_name[:8] == 'vehicle_'][:vehicles_n]
    # person_dir_list = [join(fg_root_dir, dir_name) for dir_name in sorted(os.listdir(fg_root_dir))\
    #     if dir_name[:7] == "person_"][:persons_n]
    # fg_dir_list = vehicle_dir_list + person_dir_list
    # data_dict_list = [
    #     read_conf(fg_dir) for fg_dir in fg_dir_list
    # ]    
    fg_dir_list = []
    category_list = ["vehicle", "object", "bicycle", "motorcycle", "person"]      # FIXME: We do not care persons!
    nums_list = [vehicles_n, objects_n, bicycles_n, motorcycles_n, persons_n]      # FIXME: We do not care persons!
    for category, num in zip(category_list, nums_list):
        fg_dir_list = fg_dir_list + [join(fg_root_dir, dir_name) for dir_name in sorted(os.listdir(fg_root_dir))\
         if '%s_' % category in dir_name][:num]
    data_dict_list = [
        read_conf(fg_dir) for fg_dir in fg_dir_list
    ]    

    # Process.
    n_threads = args.n_threads
    total_idx_list = list(range(n_images))
    
    for instance_idx in range(instances_n):
        # Args.
        data_dict = data_dict_list[instance_idx]
        world_coord_list = data_dict["world_coord_list"]
        theta_list = data_dict["base_angle_list"]
        mesh_bias = data_dict["mesh_bias"]
        pitch_angle = data_dict['pitch_angle']
        yaw_angle = data_dict["yaw_angle"]
        mesh_path = data_dict["mesh_dir"]
        mode = data_dict.get("mode", "NeuS")
        category = data_dict.get("category", "vehicle")
        
        # Read the mesh.
        if mode == "SNeRF":
            mesh_points, center = read_obj(mesh_path, 1)    
        else:
            mesh_points, center = read_ply(mesh_path, 1)
        
        process_list = []
        for id in range(n_threads):
            step = n_images//n_threads
            partial_list = total_idx_list[id * step : (id+1) * step]
            if id == (n_threads-1): partial_list = total_idx_list[id * step :]
            one_process = multiprocessing.Process(target=generate_one_shadow, kwargs={
                "mesh_points": mesh_points,
                "center": center,
                "idx_list": partial_list,
                "instance_idx": instance_idx,
                "dir": dir,
                "im_dir": im_dir,
                "out_dir": out_dir,
                "world_coord_list": world_coord_list,
                "theta_list": theta_list,
                "mesh_bias": mesh_bias,
                "pitch_angle": pitch_angle,
                "yaw_angle": yaw_angle,
                "check": check,
                "align": align
            })
            one_process.start()
            process_list.append(one_process)
            print('Process %d starts.' % id)                
        # while one_process.is_alive():
        #     time.sleep(.1)
        for process in process_list:
            process.join()
        
        print('Instance %d finish.' % instance_idx)
