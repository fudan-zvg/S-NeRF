import os
import argparse
import threading

def prep_depth():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--sample_token", type=str)
    arg_parser.add_argument("--sweeps_n", type=int) 
    arg_parser.add_argument("--model", type=str, default="SDC") # "NLSPN" or "FDCC" or "SDC"
    args = arg_parser.parse_args()

    GPU_LIST = ['0', '1']
    CHANNEL_LIST = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    GPU_num_per_channel = 1
    sample_token = args.sample_token
    sweeps_n = args.sweeps_n
    depth_completion_mode = args.model

    part_num = len(CHANNEL_LIST) // (len(GPU_LIST) // GPU_num_per_channel)
    th_per_part = len(GPU_LIST) // GPU_num_per_channel

    for part_idx in range(part_num):
        th_list = []
        for th_idx in range(th_per_part):
            gpu_used = ','.join(GPU_LIST[(th_idx * GPU_num_per_channel) : ((th_idx+1) * GPU_num_per_channel)])
            gpu_for_NLSPN = gpu_used[-1]
            # import pdb; pdb.set_trace()
            cmd = "CUDA_VISIBLE_DEVICES=%s python -u \
                scripts/YORO_1CAM_PIPELINE_REFINE.py --channel %s --sample %s --sweeps_n %d --gpu_for_NLSPN %s --depth_completion_mode %s" % \
                (gpu_used, CHANNEL_LIST[part_idx*th_per_part+th_idx], sample_token, sweeps_n, gpu_for_NLSPN, depth_completion_mode)
            th = threading.Thread(target=os.system, args=(cmd, ))
            th.start()
            th_list.append(th)
            
        for th in th_list:
            th.join()

if __name__ == "__main__":
    prep_depth()   