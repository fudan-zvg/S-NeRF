import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch
from PIL import Image
import random
import torchvision.transforms.functional as F
from Utils.utils import depth_read

class My_Dataset(Dataset):
    
    def __init__(self,
                 input_path):
        
        self.input_path = input_path
        
        self.lidar_in_dir = os.path.join(self.input_path, "gt")
        self.rgb_in_dir = os.path.join(self.input_path, "img")
        # self.gt_in_dir = os.path.join(self.input_path, "Gt")
        
        self.lidar_in_path_list = [os.path.join(self.lidar_in_dir, filename) for filename in sorted(os.listdir(self.lidar_in_dir))]
        self.rgb_in_path_list = [os.path.join(self.rgb_in_dir, filename) for filename in sorted(os.listdir(self.rgb_in_dir))]
        # self.gt_in_path_list = [os.path.join(self.gt_in_dir, filename) for filename in sorted(os.listdir(self.gt_in_dir))]
        
        if not (len(self.lidar_in_path_list) == len(self.rgb_in_path_list)):
            print("The file's num is not consistant!")
            print("Depth: %d RGB: %d" % (len(self.lidar_in_path_list), len(self.rgb_in_path_list)))
            print("Use the least num as n_images.")
                
        self.n_images = min([len(self.lidar_in_path_list), len(self.rgb_in_path_list)])
        print("Totally %d files in dataset." % (self.n_images))
        
    def __len__(self):
        return self.n_images
    
    def __getitem__(self, idx):
        return self.lidar_in_path_list[idx], self.rgb_in_path_list[idx]