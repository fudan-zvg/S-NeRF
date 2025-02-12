import os, argparse
import time
import random
import numpy as np
from tqdm import tqdm
# import mmcv
from omegaconf import OmegaConf


'''
Note that we assume the bg already exists.
'''
def random_choose_scan(scan_list, size):
    if scan_list == []: return ""
    scan_chosen_list = np.random.choice(scan_list, size=size, replace=False).tolist()
    scan_id = ', '.join(scan_chosen_list)
    scan_id = '\"' + scan_id + '\"'
    return scan_id

def run_pipeline(args):
    vehicles_n = args['vehicles_n']
    persons_n = args['persons_n']
    objects_n = args['objects_n']
    bicycles_n = args['bicycles_n']
    motorcycles_n = args['motorcycles_n']
    n_images = args["n_images"]
    scene_name = args["scene_name"]
    gpus = args['gpus']
    render_factor = 1
    result_name = args["result_name"]

    n_threads = 8
    wkdir = args["wkdir"]
    seed_val = random.randint(0, 255)
    
    bg_already_exists = args["bg_already_exists"]  
    only_side_cam = args["only_side_cam"]
    only_front_cam = args["only_front_cam"]
    no_added_vehicles = args["no_added_vehicles"]

    flag = 0

    if not bg_already_exists:
        flag = os.system("python api_code/background_zipnerf.py --Part_num %d --render_factor %d\
                        --gpus %s --result_name %s --RENDER_N %d --seed_val %d --scene_name %s --mode %s --wkdir %s %s %s" \
                         % (1, render_factor, gpus, result_name, n_images, seed_val, scene_name, "full_set", wkdir, \
                            '' if not only_front_cam else '--only_front_cam',
                            '' if not only_side_cam else '--only_side_cam'))

    if only_side_cam or no_added_vehicles:
        if flag == 0:
            flag = os.system("python annotate_code/get_results_only_side_cam.py --bg_name %s --wkdir %s" % (result_name, wkdir))        

    else:
        # 2. Get the foreground instances.
        if flag == 0:    # Get positions.
            flag = os.system("CUDA_VISIBLE_DEVICES=%s python stage0_code/generate_calibs.py --bg_name %s \
                            --vehicles_n %d --persons_n %d --objects_n %d --bicycles_n %d --motorcycles_n %d \
                            --render_factor %d --n_images %d --wkdir %s --mesh_render %d"
                            % (gpus, result_name, vehicles_n, persons_n, objects_n, bicycles_n, motorcycles_n, render_factor, -1, wkdir,
                               args['mesh_render'])) ### debug for mesh render # 0 for neus 1 for mesh render

        # ############################################
        # Note that the intrinsics have been scaled! 
        # ############################################
        
        if flag == 0:    # vehicles.
            flag = os.system("python api_code/mesh_api.py --meshes_n %d --gpus %s --bg_name %s "
                             "--render_factor %d --wkdir %s --category vehicle" \
                            % (vehicles_n, gpus, result_name, render_factor, wkdir))
        if flag == 0:    # persons.
            flag = os.system("python api_code/mesh_api.py --meshes_n %d --gpus %s --bg_name %s --render_factor %d --wkdir %s --category person" \
                            % (persons_n, gpus, result_name, render_factor, wkdir))

        if flag == 0:    # bicycles.
            flag = os.system("python api_code/mesh_api.py --meshes_n %d --gpus %s --bg_name %s --render_factor %d --wkdir %s --category bicycle" \
                            % (bicycles_n, gpus, result_name, render_factor, wkdir))
        if flag == 0:    # motorcycles.
            flag = os.system("python api_code/mesh_api.py --meshes_n %d --gpus %s --bg_name %s --render_factor %d --wkdir %s --category motorcycle" \
                            % (motorcycles_n, gpus, result_name, render_factor, wkdir))
        if flag == 0:
            flag = os.system("CUDA_VISIBLE_DEVICES=%s python stage1_code/generate_images.py --bg_name %s --vehicles_n %d --persons_n %d --objects_n %d --bicycles_n %d --motorcycles_n %d\
                            --render_factor %d --n_threads %d --n_images %d --wkdir %s"\
                            % (gpus, result_name, vehicles_n, persons_n, objects_n, bicycles_n, motorcycles_n, render_factor, n_threads, -1, wkdir))

        if flag == 0:
            flag = os.system("CUDA_VISIBLE_DEVICES=%s python stage2_code/inpainting_fig.py --wkdir %s" % (gpus, wkdir))
            

        if flag == 0:
            flag = os.system("python stage3_code/mesh_shadow.py --bg_name %s --vehicles_n %d --persons_n %d --objects_n %d --bicycles_n %d --motorcycles_n %d\
                            --n_threads %d --wkdir %s"\
                            % (result_name, vehicles_n, persons_n, objects_n, bicycles_n, motorcycles_n, n_threads, wkdir))
            
        if flag == 0:
            flag = os.system("python annotate_code/get_results.py --bg_name %s --wkdir %s" % (result_name, wkdir))  
            
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument('--gpu', required=True,
                        help='use which gpu') # also assign workdir wkdir=wkdir_{gpu}
    parser.add_argument('--n_images', type=int, default=0,
                        help='use which gpu') # also assign workdir wkdir=wkdir_{gpu}
    args_config = parser.parse_args()

    cfg = OmegaConf.load(args_config.config)

    os.makedirs('annotation',exist_ok=True)
    if args_config.n_images > 0 :
        cfg.n_images = args_config.n_images

    #### here to add scenes
    # result_name_list = ['0029075']
    result_name_list = os.listdir('./zipnerf/ckpt')

    args = {}
    
    #######################################################
    args["gpus"] = "\"{}\"".format(args_config.gpu)
    os.makedirs(f'wkdir_{args_config.gpu}', exist_ok=True)
    args["wkdir"] = "wkdir_{}".format(args_config.gpu)                             #              # Be very careful.
    #######################################################
    args["n_images"] = cfg.n_images
    args["bg_already_exists"] = cfg.bg_already_exists
    args["only_side_cam"] = cfg.only_side_cam
    args["only_front_cam"] = cfg.only_front_cam
    args["no_added_vehicles"] = cfg.no_added_vehicles


    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    
    for result_name in tqdm(result_name_list):
        args["result_name"] = result_name
        args["scene_name"] = result_name[-7:]
        args["vehicles_n"] = cfg.vehicles_n
        args["persons_n"] = cfg.persons_n
        args["objects_n"] = cfg.objects_n
        args["bicycles_n"] = cfg.bicycles_n
        args["motorcycles_n"] = cfg.motorcycles_n
        args['mesh_render'] = 1
        args['only_render_bg'] = cfg.get('only_render_bg', False)

        run_pipeline(args)