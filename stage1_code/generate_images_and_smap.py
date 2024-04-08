'''
Generate depth map, bound, fuse results, mask data.
'''

import os
import glob
import yaml
import argparse
import numpy as np
import multiprocessing
from os.path import join
from math import atan2, pi, sin, cos

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
    create_dir(join(out_dir, "mask"))
    create_dir(join(out_dir, "fuse"))

    for vehicle_idx in range(n_vehicles):
        create_dir(join(out_dir, "bound_%d" % vehicle_idx))
        create_dir(join(out_dir, "depth_%d" % vehicle_idx))
        create_dir(join(out_dir, "mask_%d" % vehicle_idx))


def read_conf(fg_dir):
    conf_file = join(fg_dir, "meta_data.yaml")
    with open(conf_file, 'r', encoding='utf-8') as f:
        conf = f.read()
        yamlconfig = yaml.load(conf, Loader=yaml.FullLoader)
    return yamlconfig

def fuse_sem(sem, mask, car_ind):
    return (1 - mask[..., 0] / 255) * sem + mask[..., 0] / 255 * car_ind

def main(out_dir,
         bg_dir,
         idx_list,
         rgb_file,
         sem_file,
         fg_data_dict):
    #### Process. ####
    n_vehicles = len(fg_data_dict)

    # Process.
    for idx in idx_list:

        total_fuse_im = None
        total_bound_im = None
        total_mask_im = None
        total_depth_im = None
        total_fuse_sem = None

        for vehicle_idx in range(n_vehicles):

            # Read the foreground data.
            data_dict = fg_data_dict[vehicle_idx]
            fg_dir = data_dict['fg_dir']
            world_coord_list = data_dict['world_coord_list']
            base_angle_list = data_dict['base_angle_list']
            mesh_dir = data_dict['mesh_dir']

            #### Calculate the moving vehicle ####
            world_coord = world_coord_list[idx]
            base_angle = base_angle_list[idx]

            ##########################################################################
            plane_coord, im = find_position_rgb(rgb_file,
                                                bg_dir,
                                                idx,
                                                world_coord)



            # angle = find_angle(bg_dir,
            #                 idx,
            #                 base_angle)

            # calib, P = find_camcalib(bg_dir,
            #                         idx,
            #                         world_coord,
            #                         angle)

            calib, P = get_camcalib(bg_dir,
                                    idx,
                                    world_coord,
                                    base_angle)

            inverse_flag = False if vehicle_idx != 1 else True
            # inverse_flag = False
            sc = cal_sc(idx, mesh_dir, fg_dir, bg_dir, world_coord, base_angle, inverse_flag)
            # If not want to use calculated sc, let sc = None.
            vehicle, mask_im, depth_im = decode_image(fg_dir, idx, P, plane_coord, inverse_flag, sc)

            _, j_list, _ = np.where(np.array(mask_im) > 0)
            if j_list.size == 0:
                r = 1
            else:
                mask_length = np.max(j_list) - np.min(j_list)
                r = int((mask_length / 50) ** .82)

            total_fuse_im, bound_im = fuse(im if not total_fuse_im else total_fuse_im,
                                           vehicle,
                                           mask_im,
                                           r)

            sem = get_im(sem_file, idx)
            car_ind = 2  # the semantic label of car
            total_fuse_sem = fuse_sem(sem if not total_fuse_sem else total_fuse_sem, mask_im, car_ind)

            # First save the individual data.
            bound_im.save(join(out_dir, "bound_%d" % vehicle_idx, '%05d.png' % idx))
            mask_im.save(join(out_dir, "mask_%d" % vehicle_idx, '%05d.png' % idx))
            depth_im.save(join(out_dir, "depth_%d" % vehicle_idx, '%05d.png' % idx))

            # Then fuse the data.
            total_mask_im = mask_im if not total_mask_im else Image.fromarray(
                (np.array(mask_im).astype(bool) | np.array(total_mask_im).astype(bool)).astype(np.uint8) * 255)
            total_bound_im = bound_im if not total_bound_im else Image.fromarray(
                (np.array(bound_im).astype(bool) | np.array(total_bound_im).astype(bool)).astype(np.uint8) * 255)
            total_depth_im = depth_im if not total_depth_im else Image.fromarray(
                np.array(depth_im) + np.array(total_depth_im))

        # Save the data.
        total_mask_im.save(join(out_dir, "mask", '%05d.png' % idx))
        total_bound_im.save(join(out_dir, "bound", '%05d.png' % idx))
        total_depth_im.save(join(out_dir, "depth", '%05d.png' % idx))
        total_fuse_im.save(join(out_dir, "fuse", '%05d.png' % idx))
        np.save(join(out_dir, "semantic","%05d.npy" % idx), total_fuse_sem)

        print("Finish: %d." % idx)


if __name__ == '__main__':

    root_dir = '.'
    #### Args. ####

    # Background and base.
    out_dir = join(root_dir, 'out_dir_stage1')
    bg_dir = join(root_dir, 'raw_data', 'background', 'test_semantic')
    n_images = 200
    rgb_file = np.load(join(bg_dir, 'rgb.npy'))
    sem_file = np.load(join(bg_dir), 'semantic.npy')

    # For foreground data.
    fg_dir_list = [
        join(root_dir, 'raw_data', 'foreground', '7'),
        join(root_dir, 'raw_data', 'foreground', 'nerf_data_moving2_7')
    ]
    fg_data_dict = [
        read_conf(fg_dir) for fg_dir in fg_dir_list
    ]

    ################

    # Make dirs.
    n_vehicles = len(fg_data_dict)
    create_dirs(n_vehicles)

    # Copy the camera data to the out_dir.
    path = join(bg_dir, "target_poses.npy")
    out_path = join(out_dir, "target_poses.npy")
    os.system("cp %s %s" % (path, out_path))
    path = join(bg_dir, "intrinsic.npy")
    out_path = join(out_dir, "intrinsic.npy")
    os.system("cp %s %s" % (path, out_path))

    # Multiprocess.
    n_threads = 10
    total_idx_list = list(range(n_images))
    for id in range(n_threads):
        step = n_images // n_threads
        partial_list = total_idx_list[id * step: (id + 1) * step]
        if id == (n_threads - 1): partial_list = total_idx_list[id * step:]
        one_process = multiprocessing.Process(target=main, kwargs={
            "out_dir": out_dir,
            "bg_dir": bg_dir,
            "idx_list": partial_list,
            "rgb_file": rgb_file,
            "sem_file": sem_file,
            "fg_data_dict": fg_data_dict
        })
        one_process.start()
        print('Process %d starts.' % id)