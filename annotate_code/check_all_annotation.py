import os
import time
import yaml
import shutil
import argparse
import numpy as np
from os.path import join
from PIL import Image, ImageDraw
import sys
sys.path.append('.')

parser = argparse.ArgumentParser()
parser.add_argument("--delete", default=False, action="store_true")
args = parser.parse_args()

result_dir = join('./annotation')
sub_result_dirs = [sub_dir for sub_dir in sorted(os.listdir(result_dir)) if sub_dir != 'final_results']
all_ct = 0

ct_thres = 200
filtered_scenes = []

for sub_dir in sub_result_dirs:
    sub_path = join(result_dir, sub_dir)
    sub_sub_result_dirs = [_ for _ in sorted(os.listdir(sub_path)) if _ != 'merge_results']
    ct = 0
    for sub_sub_dir in sub_sub_result_dirs:
        img_path = os.path.join(sub_path, sub_sub_dir, 'image')
        sem_path = os.path.join(sub_path, sub_sub_dir, 'semantic')
        if len(os.listdir(img_path)) != len(os.listdir(sem_path)):
            print("Warning: Wrong found in %s" % os.path.join(sub_path, sub_sub_dir))    
            if args.delete:
                os.system("rm -r %s" % os.path.join(sub_path, sub_sub_dir))
            continue
        ct += len(os.listdir(img_path))
        all_ct += len(os.listdir(img_path))
    print("%s: %d" % (sub_path, ct))
    if ct <= ct_thres:
        filtered_scenes.append(os.path.basename(sub_path))
print("Total: %d" % all_ct)
print(filtered_scenes)