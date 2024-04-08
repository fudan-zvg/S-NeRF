import os
import numpy as np
import imageio
from os.path import join
from PIL import Image

out_dir = "./out_dir_stage3"

# Rendering video.
im_list = sorted(os.listdir(join(out_dir, 'image')))
im_mat_list = []
for fname in im_list:
    im_mat_list.append(np.array(Image.open(os.path.join(out_dir, 'image', fname))))

rgb_npy = np.stack(im_mat_list, axis=0)
imageio.mimsave(os.path.join(out_dir, 'video.mp4'), rgb_npy, fps=30,
                ffmpeg_params=[
                    "-crf",
                    "5"
                ])    