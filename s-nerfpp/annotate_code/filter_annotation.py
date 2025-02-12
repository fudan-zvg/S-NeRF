import os
import cv2
import multiprocessing
from tqdm import tqdm
import argparse
import numpy as np
from scipy.stats import entropy
from math import sqrt
from PIL import Image
from os.path import join
from skimage.metrics import structural_similarity as compare_ssim

'''
Note that:
This is only for segmentation,
because we just handle image and semantic.
'''


def stat_for_one_label_mat(label_mat):
    label_stat = []
    for label_idx in range(19):
        label_stat.append((label_mat == label_idx).sum())

    sum_res = np.array(label_stat).sum()

    return np.array(label_stat).max() / sum_res, label_stat[13] / sum_res


def one_process(root_dir,
                idx_list):
    
    img_dir = join(root_dir, "image")
    sem_dir = join(root_dir, "semantic")
    
    for ct, idx in enumerate(idx_list):
        f_sem_name = join(sem_dir, "%05d.png" % idx)
        f_img_name = join(img_dir, "%05d.png" % idx)
        label_mat = cv2.imread(f_sem_name)
        img_mat = cv2.imread(f_img_name)
        img_mat = cv2.cvtColor(img_mat, cv2.COLOR_BGR2RGB)
        
        if label_mat.shape != (886, 1920, 3):           # Only focus on side camera.
            continue
        
        filter_flag = False
        
        # 3. Re-Blur.
        if not filter_flag:
            blur_img_mat = cv2.GaussianBlur(img_mat, (17, 17), 0)
            score = compare_ssim(cv2.cvtColor(img_mat, cv2.COLOR_RGB2GRAY),\
                                    cv2.cvtColor(blur_img_mat, cv2.COLOR_RGB2GRAY), win_size=17)
            if score > 0.995:
                os.system("mv %s %s" % (f_img_name, join(root_dir, "filtered", "image", os.path.basename(f_img_name))))
                os.system("mv %s %s" % (f_sem_name, join(root_dir, "filtered", "semantic", os.path.basename(f_sem_name)))) 
                # print(score)
                filter_flag = True    
                
        if ct % 1000 == 0:
            print("Finish: %d." % idx)         


if __name__ == '__main__':
    
    args_parser = argparse.ArgumentParser()
    # args_parser.add_argument("--delete", action="store_true", default=False)
    args_parser.add_argument("--result_dir", type=str, required=True)
    args_parser.add_argument("--reset", action="store_true", default=False)
    args_parser.add_argument("--n_workers", type=int, default=16)
    args = args_parser.parse_args()


    root_dir = args.result_dir
    n_threads = args.n_workers
    img_dir = join(root_dir, "image")
    sem_dir = join(root_dir, "semantic")
    
    n_images = len(os.listdir(img_dir))
    assert len(os.listdir(img_dir)) == len(os.listdir(sem_dir))
    
    os.makedirs(join(root_dir, "filtered"), exist_ok=True)
    os.makedirs(join(root_dir, "filtered", "image"), exist_ok=True)
    os.makedirs(join(root_dir, "filtered", "semantic"), exist_ok=True)
    
    if args.reset:
        os.system("mv %s %s" % (join(root_dir, "filtered", "image", "*"), img_dir))
        os.system("mv %s %s" % (join(root_dir, "filtered", "semantic", "*"), sem_dir))

    else:
        total_idx_list = list(range(n_images))
        threads_list = []
        for id in range(n_threads):
            step = n_images//n_threads
            partial_list = total_idx_list[id * step : (id+1) * step]
            if id == (n_threads-1): partial_list = total_idx_list[id * step :]
            one_thread = multiprocessing.Process(target=one_process, kwargs={
                "root_dir": root_dir,
                "idx_list": partial_list
            })
            one_thread.start()
            threads_list.append(one_thread)
            
        for thread in threads_list:
            thread.join()