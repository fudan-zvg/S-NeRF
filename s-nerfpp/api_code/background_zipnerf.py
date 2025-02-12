# Run the code under the root dir.
import os
import glob
import shutil
import argparse
import imageio
import numpy as np
import threading


def scale_intrinsic(P,
                    sc):
    P[0, 0] = P[0, 0] * sc
    P[1, 1] = P[1, 1] * sc
    P[0, 2] = P[0, 2] * sc
    P[1, 2] = P[1, 2] * sc
    return P


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--scene_name', required=True, type=str)
    parser.add_argument('--Part_num', default=4, type=int)
    parser.add_argument('--gpus', type=str, default="0, 1, 2, 3")
    parser.add_argument('--render_factor', default=4, type=int)
    parser.add_argument('--result_name', type=str, required=True)
    parser.add_argument('--RENDER_N', type=int, default=1000)
    parser.add_argument('--seed_val', type=int, default=21)
    parser.add_argument('--mode', type=str, default="full_set")
    parser.add_argument('--wkdir', type=str, default="wkdir_0", required=True)
    parser.add_argument('--only_side_cam', action="store_true", default=False)
    parser.add_argument('--only_front_cam', action="store_true", default=False)
    parser.add_argument('--demo', default=0, type=int)

    args = parser.parse_args()

    root_dir = os.path.abspath('.')
    wkdir = args.wkdir
    # os.chdir("./DirectVoxGO")
    os.chdir("./zipnerf")

    result_name = args.result_name
    # out_dir = "%s/raw_data/background/%s" % (root_dir, result_name)
    out_dir = os.path.join(root_dir, wkdir, "raw_data", "background", result_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print("Remove all files under %s" % out_dir)
        # raise FileExistsError
        f_list = glob.glob(os.path.join(out_dir, '*'))
        for f in f_list:
            if os.path.isdir(f):
                shutil.rmtree(f)      
            else:      
                os.remove(f)
                
    Part_num = args.Part_num
    GPU_list = args.gpus.split(', ')
    render_factor = args.render_factor
    RENDER_N = args.RENDER_N
    chunk = 1024 * 32
    net_chunk = 1024 * 32
    seed_val = args.seed_val
    scene_name = args.scene_name
    mode = args.mode
    only_side_cam = args.only_side_cam
    only_front_cam = args.only_front_cam


    which_python = 'python'
    th_list = []
    for idx in range(1, Part_num + 1):
        exp_name = 'ckpt/{}'.format(scene_name)
        data_dir = '../dataset/processed_dataset/{}'.format(scene_name)
        cmd = """CUDA_VISIBLE_DEVICES={} {} random_render_waymo_seq.py \
                --gin_configs=configs/waymo.gin \
                --gin_bindings="Config.data_dir = '{}'" \
                --gin_bindings="Config.exp_name = '{}'" \
                --gin_bindings="Config.sample_n_test = 7" \
                --gin_bindings="Config.sample_m_test = 3" \
                --gin_bindings="Config.render_path = False" \
                --gin_bindings="Config.only_side_cam = {}" \
                --gin_bindings="Config.only_front_cam = {}" \
                --gin_bindings="Config.root_dir = '{}'" \
                --gin_bindings="Config.RENDER_N = {}" \
                --gin_bindings="Config.wkdir = '{}'" \
                --gin_bindings="Config.result_name = '{}'"\
                --gin_bindings="Config.demo = '{}'"\
                """.\
            format(GPU_list[idx - 1], which_python, data_dir, exp_name, only_side_cam, only_front_cam, root_dir,
               RENDER_N, wkdir, result_name, args.demo)

        os.system(cmd)

    # Scale the intrinsic.
    intrinsic = np.load(os.path.join(out_dir, "intrinsic.npy"))
    intrinsic = scale_intrinsic(intrinsic, sc=1 / render_factor)
    np.save(os.path.join(out_dir, "intrinsic.npy"), intrinsic)
    
    # Make the dir.
    os.makedirs(os.path.join(out_dir, "sample_vis"))
