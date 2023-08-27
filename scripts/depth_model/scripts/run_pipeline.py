import os
import argparse
import threading

CHANNEL_LIST = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

def prep_depth():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--scene_name", type=str)
    arg_parser.add_argument("--gpus", type=list, default=['0','1','2','3','4','5'])
    arg_parser.add_argument("--model", type=str, default="SDC") # "NLSPN" or "FDCC" or "SDC"
    args = arg_parser.parse_args()

    GPU_LIST = args.gpus[:len(CHANNEL_LIST)]
    GPU_num_per_channel = 1
    depth_completion_mode = args.model

    part_num = len(CHANNEL_LIST) // (len(GPU_LIST) // GPU_num_per_channel)
    th_per_part = len(GPU_LIST) // GPU_num_per_channel

    for part_idx in range(part_num):
        th_list = []
        for th_idx in range(th_per_part):
            gpu_used = ','.join(GPU_LIST[(th_idx * GPU_num_per_channel) : ((th_idx+1) * GPU_num_per_channel)])
            gpu_for_NLSPN = gpu_used[-1]
            # import pdb; pdb.set_trace()
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_used} python -u scripts/YORO_1CAM_PIPELINE_REFINE.py \
                    --channel {CHANNEL_LIST[part_idx*th_per_part+th_idx]} \
                    --scene_name {args.scene_name} \
                    --gpu_for_NLSPN {gpu_for_NLSPN} \
                    --depth_completion_mode {depth_completion_mode}"
            
            th = threading.Thread(target=os.system, args=(cmd, ))
            th.start()
            th_list.append(th)
            
        for th in th_list:
            th.join()

if __name__ == "__main__":
    prep_depth()   