import os
import glob
import shutil
import argparse
import numpy as np
from os.path import join

def get_person_ckpt():
    person_root = join(simnerf_root, 'TEXTure_ckpt/person_temp')
    obj_pix = ['jumpobj', 'runobj', 'walkobj']
    persons = os.listdir(join(person_root, 'persons'))
    persons = sorted([person for person in persons if person != 'static' and person != 'readme.md'])
    ckpts = []
    for person in persons:
        for mode in obj_pix:
            obj_files = os.listdir(join(person_root, 'persons', person, mode))
            obj_files = sorted([file for file in obj_files if file[-3:] == 'obj'])
            obj_files = sorted([join(person_root, 'persons', person, mode, file) for file in obj_files])
            ckpts = ckpts+obj_files
    return ckpts

def get_class_ckpt(class_name='bicycle'):
    class_root = join(simnerf_root, 'TEXTure_ckpt/textured_mesh')
    objs = sorted(os.listdir(join(class_root, class_name)))
    ckpts = [join(class_root,class_name,obj,'mesh/mesh.obj') for obj in objs]
    ckpts = [ckpt for ckpt in ckpts if os.path.exists(ckpt)]
    return ckpts




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--meshes_n', type=int, default=1)
    parser.add_argument('--render_factor', type=int, default=4)
    parser.add_argument('--gpus', type=str, default="0, 1, 2, 3")
    parser.add_argument('--bg_name', type=str, default='test_random_bg')
    parser.add_argument('--multi_process', default=False, action="store_true")
    parser.add_argument('--wkdir', type=str, default="wkdir_0", required=True)
    parser.add_argument('--category', type=str, default="vehicle")

    args = parser.parse_args()

    simnerf_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    wkdir = args.wkdir
    GPU_list = args.gpus.split(', ')
    which_python = 'python'

    cam_path = os.path.abspath(join(simnerf_root, wkdir, 'raw_data', 'background', args.bg_name, 'target_poses.npy'))
    intrinsic_path = os.path.abspath(join(simnerf_root, wkdir, 'raw_data', 'background', args.bg_name, 'intrinsic.npy'))


    ckpt_dir = join(simnerf_root, 'TEXTure_ckpt','textured_mesh')

    if args.category == 'vehicle':
        base = join(ckpt_dir, 'car',)
        ckpts = sorted(os.listdir(base))
        checkpoints = [join(base, a,'mesh','mesh.obj') for a in ckpts]
        checkpoints = [checkpoint for checkpoint in checkpoints if os.path.exists(checkpoint)]
        # checkpoints = []   ### todo: debug
        root_neus = join(simnerf_root, 'TEXTure_ckpt','align_meshes')
        ckpts_neus = [join(root_neus, a) for a in os.listdir(root_neus)]
        checkpoints = checkpoints+ckpts_neus

        print(f"{len(checkpoints)} cars")

    if args.category == 'person':
        checkpoints = get_person_ckpt()
    elif args.category != 'vehicle':
        checkpoints = get_class_ckpt(args.category)

    ckpts_chosen = np.random.choice(checkpoints, size=args.meshes_n, replace=True if args.meshes_n>len(checkpoints) else False).tolist() ### should be false
    for i in range(args.meshes_n):
        meta_path = os.path.abspath(join(simnerf_root, wkdir, 'raw_data', 'foreground', "{}_{}".format(args.category,i), 'meta_data.yaml'))
        save_path = os.path.abspath(join(simnerf_root, wkdir, 'raw_data', 'foreground', "{}_{}".format(args.category,i)))
        f_list = glob.glob(join(save_path, '*'))
        for f in f_list:
            if os.path.basename(f) == "image" or os.path.basename(f) == "mask":
                shutil.rmtree(f)
            elif os.path.basename(f) == "mesh.ply":
                os.remove(f)        
        ckpt = ckpts_chosen[i]
        with open(join(save_path,'checkpoint.txt'), 'w') as file:
            file.write(ckpt)
        cmd = "CUDA_VISIBLE_DEVICES={} {} mesh_renderer.py --ckpt_path {} --camera_path {} --intrinsic_path {} --meta_path {} --save_path {}" \
              " --render_factor {} --category {}".\
            format(GPU_list[0], which_python, ckpt, cam_path, intrinsic_path, meta_path,
                   save_path, args.render_factor, args.category)
        os.chdir(join(simnerf_root, 'api_code'))
        os.system(cmd)
        os.chdir('..')





