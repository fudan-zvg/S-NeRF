import enum
import os
import torch
import numpy as np
from PIL import ExifTags
from PIL import Image
import collections
import random


class Timing:
    """
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        print(self.name, "elapsed", self.start.elapsed_time(self.end), "ms")


def nan_sum(x):
    return torch.isnan(x).sum()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class DataSplit(enum.Enum):
    """Dataset split."""
    TRAIN = 'train'
    TEST = 'test'


class BatchingMethod(enum.Enum):
    """Draw rays randomly from a single image or all images, in each batch."""
    ALL_IMAGES = 'all_images'
    SINGLE_IMAGE = 'single_image'


def open_file(pth, mode='r'):
    return open(pth, mode=mode)


def file_exists(pth):
    return os.path.exists(pth)


def listdir(pth):
    return os.listdir(pth)


def isdir(pth):
    return os.path.isdir(pth)


def makedirs(pth):
    os.makedirs(pth, exist_ok=True)


def load_img(pth):
    """Load an image and cast to float32."""
    with open_file(pth, 'rb') as f:
        image = np.array(Image.open(f), dtype=np.float32)
    return image


def load_exif(pth):
    """Load EXIF data for an image."""
    with open_file(pth, 'rb') as f:
        image_pil = Image.open(f)
        exif_pil = image_pil._getexif()  # pylint: disable=protected-access
        if exif_pil is not None:
            exif = {
                ExifTags.TAGS[k]: v for k, v in exif_pil.items() if k in ExifTags.TAGS
            }
        else:
            exif = {}
    return exif


def save_img_u8(img, pth):
    """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
    with open_file(pth, 'wb') as f:
        Image.fromarray(
            (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')


def save_img_f32(depthmap, pth):
    """Save an image (probably a depthmap) to disk as a float32 TIFF."""
    with open_file(pth, 'wb') as f:
        Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')
