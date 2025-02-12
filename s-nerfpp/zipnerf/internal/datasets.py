import abc
import copy
import json
import os
import cv2
from internal import camera_utils
from internal import configs
from internal import image as lib_image
from internal import raw_utils
from internal import utils
# from internal.camera_param import *
# from zipnerf.internal.camera_param import *
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
# This is ugly, but it works.
import sys
import imageio
import random
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, 'internal/pycolmap')
sys.path.insert(0, 'internal/pycolmap/pycolmap')
import pycolmap


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

def random_noise(x, dx):
    return random.uniform(x - dx, x + dx)


def random_interval(interval):
    a, b = interval
    a, b = min(a, b), max(a, b)
    return random.uniform(a, b)

def interpolate_two_pose(pose_0, pose_1, ratio=.5, fix_trans=False):
    '''
    We only interpolate the rotation.
    '''

    # pose_0, pose_1 = pose_0.detach().cpu().numpy(), pose_1.detach().cpu().numpy()
    pose_0, pose_1 = np.linalg.inv(pose_0), np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)

    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    if fix_trans:
        pose[:3, 3] = pose_0[:3, 3]
    else:
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    pose = np.linalg.inv(pose)

    # pose = torch.from_numpy(pose)
    return pose


def add_noise_to_pose(pose, dx, dy, dz, dx_theta, dy_theta):
    pose[0, 3] = random_noise(pose[0, 3], dx)
    pose[1, 3] = random_noise(pose[1, 3], dy)
    pose[2, 3] = random_noise(pose[2, 3], dz)

    x_theta = random_noise(0, dx_theta)
    y_theta = random_noise(0, dy_theta)

    x_rot = R.from_euler('x', x_theta, degrees=True).as_matrix()
    y_rot = R.from_euler('y', y_theta, degrees=True).as_matrix()

    pose[:3, :3] = pose[:3, :3] @ y_rot @ x_rot

    return pose


def load_dataset(split, train_dir, config):
    """Loads a split of a dataset using the data_loader specified by `config`."""
    dataset_dict = {
        'blender': Blender,
        'llff': LLFF,
        'tat_nerfpp': TanksAndTemplesNerfPP,
        'tat_fvs': TanksAndTemplesFVS,
        'dtu': DTU,
        'waymo':WAYMO,
        'nusc': NUSCENES,
        'waymo_render': WAYMO_RENDER,
        'nuscenes_render': NUSCENES_RENDER,
        'waymo_demo': WAYMO_DEMO
    }
    return dataset_dict[config.dataset_loader](split, train_dir, config)


class NeRFSceneManager(pycolmap.SceneManager):
    """COLMAP pose loader.

  Minor NeRF-specific extension to the third_party Python COLMAP loader:
  google3/third_party/py/pycolmap/scene_manager.py
  """

    def process(self):
        """Applies NeRF-specific postprocessing to the loaded pose data.

    Returns:
      a tuple [image_names, poses, pixtocam, distortion_params].
      image_names:  contains the only the basename of the images.
      poses: [N, 4, 4] array containing the camera to world matrices.
      pixtocam: [N, 3, 3] array containing the camera to pixel space matrices.
      distortion_params: mapping of distortion param name to distortion
        parameters. Cameras share intrinsics. Valid keys are k1, k2, p1 and p2.
    """

        self.load_cameras()
        self.load_images()
        # self.load_points3D()  # For now, we do not need the point cloud data.

        # Assume shared intrinsics between all cameras.
        cam = self.cameras[1]

        # Extract focal lengths and principal point parameters.
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))

        # Extract extrinsic matrices in world-to-camera format.
        imdata = self.images
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)
        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        c2w_mats = np.linalg.inv(w2c_mats)
        poses = c2w_mats[:, :3, :4]

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        names = [imdata[k].name for k in imdata]

        # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
        poses = poses @ np.diag([1, -1, -1, 1])

        # Get distortion parameters.
        type_ = cam.camera_type

        if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
            params = None
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 1 or type_ == 'PINHOLE':
            params = None
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        if type_ == 2 or type_ == 'SIMPLE_RADIAL':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 3 or type_ == 'RADIAL':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 4 or type_ == 'OPENCV':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            params['p1'] = cam.p1
            params['p2'] = cam.p2
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 5 or type_ == 'OPENCV_FISHEYE':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'k4']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            params['k3'] = cam.k3
            params['k4'] = cam.k4
            camtype = camera_utils.ProjectionType.FISHEYE

        return names, poses, pixtocam, params, camtype


def load_blender_posedata(data_dir, split=None):
    """Load poses from `transforms.json` file, as used in Blender/NGP datasets."""
    suffix = '' if split is None else f'_{split}'
    pose_file = os.path.join(data_dir, f'transforms{suffix}.json')
    with utils.open_file(pose_file, 'r') as fp:
        meta = json.load(fp)
    names = []
    poses = []
    for _, frame in enumerate(meta['frames']):
        filepath = os.path.join(data_dir, frame['file_path'])
        if utils.file_exists(filepath):
            names.append(frame['file_path'].split('/')[-1])
            poses.append(np.array(frame['transform_matrix'], dtype=np.float32))
    poses = np.stack(poses, axis=0)

    w = meta['w']
    h = meta['h']
    cx = meta['cx'] if 'cx' in meta else w / 2.
    cy = meta['cy'] if 'cy' in meta else h / 2.
    if 'fl_x' in meta:
        fx = meta['fl_x']
    else:
        fx = 0.5 * w / np.tan(0.5 * float(meta['camera_angle_x']))
    if 'fl_y' in meta:
        fy = meta['fl_y']
    else:
        fy = 0.5 * h / np.tan(0.5 * float(meta['camera_angle_y']))
    pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))
    coeffs = ['k1', 'k2', 'p1', 'p2']
    if not any([c in meta for c in coeffs]):
        params = None
    else:
        params = {c: (meta[c] if c in meta else 0.) for c in coeffs}
    camtype = camera_utils.ProjectionType.PERSPECTIVE
    return names, poses, pixtocam, params, camtype


class Dataset(torch.utils.data.Dataset):
    """Dataset Base Class.

  Base class for a NeRF dataset. Creates batches of ray and color data used for
  training or rendering a NeRF model.

  Each subclass is responsible for loading images and camera poses from disk by
  implementing the _load_renderings() method. This data is used to generate
  train and test batches of ray + color data for feeding through the NeRF model.
  The ray parameters are calculated in _generate_rays().

  The public interface mimics the behavior of a standard machine learning
  pipeline dataset provider that can provide infinite batches of data to the
  training/testing pipelines without exposing any details of how the batches are
  loaded/created or how this is parallelized. Therefore, the initializer runs
  all setup, including data loading from disk using _load_renderings(), and
  begins the thread using its parent start() method. After the initializer
  returns, the caller can request batches of data straight away.

  The internal self._queue is initialized as queue.Queue(3), so the infinite
  loop in run() will block on the call self._queue.put(self._next_fn()) once
  there are 3 elements. The main thread training job runs in a loop that pops 1
  element at a time off the front of the queue. The Dataset thread's run() loop
  will populate the queue with 3 elements, then wait until a batch has been
  removed and push one more onto the end.

  This repeats indefinitely until the main thread's training loop completes
  (typically hundreds of thousands of iterations), then the main thread will
  exit and the Dataset thread will automatically be killed since it is a daemon.

  Attributes:
    alphas: np.ndarray, optional array of alpha channel data.
    cameras: tuple summarizing all camera extrinsic/intrinsic/distortion params.
    camtoworlds: np.ndarray, a list of extrinsic camera pose matrices.
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.
    data_dir: str, location of the dataset on disk.
    disp_images: np.ndarray, optional array of disparity (inverse depth) data.
    distortion_params: dict, the camera distortion model parameters.
    exposures: optional per-image exposure value (shutter * ISO / 1000).
    far: float, far plane value for rays.
    focal: float, focal length from camera intrinsics.
    height: int, height of images.
    images: np.ndarray, array of RGB image data.
    metadata: dict, optional metadata for raw datasets.
    near: float, near plane value for rays.
    normal_images: np.ndarray, optional array of surface normal vector data.
    pixtocams: np.ndarray, one or a list of inverse intrinsic camera matrices.
    pixtocam_ndc: np.ndarray, the inverse intrinsic matrix used for NDC space.
    poses: np.ndarray, optional array of auxiliary camera pose data.
    rays: utils.Rays, ray data for every pixel in the dataset.
    render_exposures: optional list of exposure values for the render path.
    render_path: bool, indicates if a smooth camera path should be generated.
    size: int, number of images in the dataset.
    split: str, indicates if this is a "train" or "test" dataset.
    width: int, width of images.
  """

    def __init__(self,
                 split: str,
                 data_dir: str,
                 config: configs.Config):
        super().__init__()

        # Initialize attributes
        self._patch_size = np.maximum(config.patch_size, 1)
        self._batch_size = config.batch_size // config.world_size
        if self._patch_size ** 2 > self._batch_size:
            raise ValueError(f'Patch size {self._patch_size}^2 too large for ' +
                             f'per-process batch size {self._batch_size}')
        self._batching = utils.BatchingMethod(config.batching)
        self._use_tiffs = config.use_tiffs
        self._load_disps = config.compute_disp_metrics
        self._load_normals = config.compute_normal_metrics
        self._num_border_pixels_to_mask = config.num_border_pixels_to_mask
        self._apply_bayer_mask = config.apply_bayer_mask
        self._render_spherical = False

        self.local_rank = config.local_rank
        self.world_size = config.world_size
        self.split = utils.DataSplit(split)
        self.data_dir = data_dir
        self.near = config.near
        self.far = config.far
        self.render_path = config.render_path
        self.distortion_params = None
        self.disp_images = None
        self.normal_images = None
        self.alphas = None
        self.poses = None
        self.pixtocam_ndc = None
        self.metadata = None
        self.camtype = camera_utils.ProjectionType.PERSPECTIVE
        self.exposures = None
        self.render_exposures = None

        # Providing type comments for these attributes, they must be correctly
        # initialized by _load_renderings() (see docstring) in any subclass.
        self.images: np.ndarray = None
        self.camtoworlds: np.ndarray = None
        self.depths = None
        self.semantics = None
        self.masks = None
        self.local2global_idx = None
        self.num_poses = None
        self.hws = None

        self.pixtocams: np.ndarray = None
        self.height: int = None
        self.width: int = None

        # Load data from disk using provided config parameters.
        self._load_renderings(config)

        if self.render_path:
            if config.render_path_file is not None:
                with utils.open_file(config.render_path_file, 'rb') as fp:
                    render_poses = np.load(fp)
                self.camtoworlds = render_poses
            if config.render_resolution is not None:
                self.width, self.height = config.render_resolution
            if config.render_focal is not None:
                self.focal = config.render_focal
            if config.render_camtype is not None:
                if config.render_camtype == 'pano':
                    self._render_spherical = True
                else:
                    self.camtype = camera_utils.ProjectionType(config.render_camtype)

            self.distortion_params = None
            self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                       self.height)

        self._n_examples = self.camtoworlds.shape[0]

        self.cameras = (self.pixtocams,
                        self.camtoworlds,
                        self.distortion_params,
                        self.pixtocam_ndc)

        # Seed the queue with one batch to avoid race condition.
        if self.split == utils.DataSplit.TRAIN:
            self._next_fn = self.next_train
        else:
            self._next_fn = self._next_test

    def next_train(self, item):
        if self._patch_size == 1:
            return self._next_train(self._batch_size, self._patch_size)
        else:
            patch_batch = self._batch_size//4
            pix_batch = self._batch_size-patch_batch
            pat = self._next_train(patch_batch, self._patch_size)
            pix = self._next_train(pix_batch, 1)
            batch = {}
            for key in pat.keys():
                if pat[key] is None:
                    batch[key] = None
                else:
                    shape = [*pat[key].shape]
                    shape[0] = -1
                    batch[key] = torch.cat([pat[key], pix[key].reshape(*shape)], 0)

            return batch

    @property
    def size(self):
        return self._n_examples

    def __len__(self):
        return self._n_examples

    @abc.abstractmethod
    def _load_renderings(self, config):
        """Load images and poses from disk.

    Args:
      config: utils.Config, user-specified config parameters.
    In inherited classes, this method must set the following public attributes:
      images: [N, height, width, 3] array for RGB images.
      disp_images: [N, height, width] array for depth data (optional).
      normal_images: [N, height, width, 3] array for normals (optional).
      camtoworlds: [N, 3, 4] array of extrinsic pose matrices.
      poses: [..., 3, 4] array of auxiliary pose data (optional).
      pixtocams: [N, 3, 4] array of inverse intrinsic matrices.
      distortion_params: dict, camera lens distortion model parameters.
      height: int, height of images.
      width: int, width of images.
      focal: float, focal length to use for ideal pinhole rendering.
    """

    def _make_ray_batch(self,
                        pix_x_int,
                        pix_y_int,
                        cam_idx,
                        lossmult=None
                        ):
        """Creates ray data batch from pixel coordinates and camera indices.

    All arguments must have broadcastable shapes. If the arguments together
    broadcast to a shape [a, b, c, ..., z] then the returned utils.Rays object
    will have array attributes with shape [a, b, c, ..., z, N], where N=3 for
    3D vectors and N=1 for per-ray scalar attributes.

    Args:
      pix_x_int: int array, x coordinates of image pixels.
      pix_y_int: int array, y coordinates of image pixels.
      cam_idx: int or int array, camera indices.
      lossmult: float array, weight to apply to each ray when computing loss fn.

    Returns:
      A dict mapping from strings utils.Rays or arrays of image data.
      This is the batch provided for one NeRF train or test iteration.
    """

        broadcast_scalar = lambda x: np.broadcast_to(x, pix_x_int.shape)[..., None]
        ray_kwargs = {
            'lossmult': broadcast_scalar(1.) if lossmult is None else lossmult,
            'near': broadcast_scalar(self.near),
            'far': broadcast_scalar(self.far),
            'cam_idx': broadcast_scalar(cam_idx),
        }
        # Collect per-camera information needed for each ray.
        if self.metadata is not None:
            # Exposure index and relative shutter speed, needed for RawNeRF.
            for key in ['exposure_idx', 'exposure_values']:
                idx = 0 if self.render_path else cam_idx
                ray_kwargs[key] = broadcast_scalar(self.metadata[key][idx])
        if self.exposures is not None:
            idx = 0 if self.render_path else cam_idx
            ray_kwargs['exposure_values'] = broadcast_scalar(self.exposures[idx])
        if self.render_path and self.render_exposures is not None:
            ray_kwargs['exposure_values'] = broadcast_scalar(
                self.render_exposures[cam_idx])

        pixels = dict(pix_x_int=pix_x_int, pix_y_int=pix_y_int, **ray_kwargs)

        # Slow path, do ray computation using numpy (on CPU).
        batch = camera_utils.cast_ray_batch(self.cameras, pixels, self.camtype)

        if not self.render_path and self.images is not None:
            batch['rgb'] = self.images[cam_idx, pix_y_int, pix_x_int]
        # if self._load_disps:
        if self.depths is not None:
            batch['depth'] = self.depths[cam_idx, pix_y_int, pix_x_int]
        if self.semantics is not None:
            batch['semantic'] = self.semantics[cam_idx, pix_y_int, pix_x_int]
        if self.masks is not None:
            batch['mask'] = self.masks[cam_idx, pix_y_int, pix_x_int]
        if self.local2global_idx is not None:
            batch['glo_idx'] = broadcast_scalar(self.local2global_idx[cam_idx])

        if self._load_normals:
            batch['normals'] = self.normal_images[cam_idx, pix_y_int, pix_x_int]
            batch['alphas'] = self.alphas[cam_idx, pix_y_int, pix_x_int]
        return {k: torch.from_numpy(v.copy()).float() if v is not None else None for k, v in batch.items()}

    def _next_train(self, batch_size, patch_size):
        """Sample next training batch (random rays)."""
        # We assume all images in the dataset are the same resolution, so we can use
        # the same width/height for sampling all pixels coordinates in the batch.
        # Batch/patch sampling parameters.
        num_patches = batch_size // patch_size ** 2
        lower_border = self._num_border_pixels_to_mask
        upper_border = self._num_border_pixels_to_mask + patch_size - 1
        # Random pixel patch x-coordinates.
        pix_x_int = np.random.randint(lower_border, self.width - upper_border,
                                      (num_patches, 1, 1))
        # Random pixel patch y-coordinates.
        pix_y_int = np.random.randint(lower_border, self.height - upper_border,
                                      (num_patches, 1, 1))
        # Add patch coordinate offsets.
        # Shape will broadcast to (num_patches, _patch_size, _patch_size).
        patch_dx_int, patch_dy_int = camera_utils.pixel_coordinates(
            patch_size, patch_size)
        pix_x_int = pix_x_int + patch_dx_int
        pix_y_int = pix_y_int + patch_dy_int
        # Random camera indices.
        if self._batching == utils.BatchingMethod.ALL_IMAGES:
            cam_idx = np.random.randint(0, self._n_examples, (num_patches, 1, 1))
        else:
            cam_idx = np.random.randint(0, self._n_examples, (1,))

        if self._apply_bayer_mask:
            # Compute the Bayer mosaic mask for each pixel in the batch.
            lossmult = raw_utils.pixels_to_bayer_mask(pix_x_int, pix_y_int)
        else:
            lossmult = None

        return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx,
                                    lossmult=lossmult)

    def generate_ray_batch(self, cam_idx: int):
        """Generate ray batch for a specified camera in the dataset."""
        if self._render_spherical:
            camtoworld = self.camtoworlds[cam_idx]
            rays = camera_utils.cast_spherical_rays(
                camtoworld, self.height, self.width, self.near, self.far)
            return rays
        else:
            # Generate rays for all pixels in the image.
            height, width = self.hws[cam_idx] if self.hws is not None else (self.height, self.width)
            pix_x_int, pix_y_int = camera_utils.pixel_coordinates(
                width, height)
            return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx)

    def _next_test(self, item):
        """Sample next test batch (one full image)."""
        return self.generate_ray_batch(item)

    def collate_fn(self, item):
        return self._next_fn(item[0])

    def __getitem__(self, item):
        return self._next_fn(item)


class Blender(Dataset):
    """Blender Dataset."""

    def _load_renderings(self, config):
        """Load images from disk."""
        if config.render_path:
            raise ValueError('render_path cannot be used for the blender dataset.')
        pose_file = os.path.join(self.data_dir, f'transforms_{self.split.value}.json')
        with utils.open_file(pose_file, 'r') as fp:
            meta = json.load(fp)
        images = []
        disp_images = []
        normal_images = []
        cams = []
        for _, frame in enumerate(meta['frames']):
            fprefix = os.path.join(self.data_dir, frame['file_path'])

            def get_img(f, fprefix=fprefix):
                image = utils.load_img(fprefix + f)
                if config.factor > 1:
                    image = lib_image.downsample(image, config.factor)
                return image

            if self._use_tiffs:
                channels = [get_img(f'_{ch}.tiff') for ch in ['R', 'G', 'B', 'A']]
                # Convert image to sRGB color space.
                image = lib_image.linear_to_srgb_np(np.stack(channels, axis=-1))
            else:
                image = get_img('.png') / 255.
            images.append(image)

            if self._load_disps:
                disp_image = get_img('_disp.tiff')
                disp_images.append(disp_image)
            if self._load_normals:
                normal_image = get_img('_normal.png')[..., :3] * 2. / 255. - 1.
                normal_images.append(normal_image)

            cams.append(np.array(frame['transform_matrix'], dtype=np.float32))

        self.images = np.stack(images, axis=0)
        if self._load_disps:
            self.disp_images = np.stack(disp_images, axis=0)
        if self._load_normals:
            self.normal_images = np.stack(normal_images, axis=0)
            self.alphas = self.images[..., -1]

        rgb, alpha = self.images[..., :3], self.images[..., -1:]
        self.images = rgb * alpha + (1. - alpha)  # Use a white background.
        self.height, self.width = self.images.shape[1:3]
        self.camtoworlds = np.stack(cams, axis=0)
        self.focal = .5 * self.width / np.tan(.5 * float(meta['camera_angle_x']))
        self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                   self.height)


class LLFF(Dataset):
    """LLFF Dataset."""

    def _load_renderings(self, config):
        """Load images from disk."""
        # Set up scaling factor.
        image_dir_suffix = ''
        # Use downsampling factor (unless loading training split for raw dataset,
        # we train raw at full resolution because of the Bayer mosaic pattern).
        if config.factor > 0 and not (config.rawnerf_mode and
                                      self.split == utils.DataSplit.TRAIN):
            image_dir_suffix = f'_{config.factor}'
            factor = config.factor
        else:
            factor = 1

        # Copy COLMAP data to local disk for faster loading.
        colmap_dir = os.path.join(self.data_dir, 'sparse/0/')

        # Load poses.
        if utils.file_exists(colmap_dir):
            pose_data = NeRFSceneManager(colmap_dir).process()
        else:
            # Attempt to load Blender/NGP format if COLMAP data not present.
            pose_data = load_blender_posedata(self.data_dir)
        image_names, poses, pixtocam, distortion_params, camtype = pose_data

        # Previous NeRF results were generated with images sorted by filename,
        # use this flag to ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        poses = poses[inds]

        # Load bounds if possible (only used in forward facing scenes).
        posefile = os.path.join(self.data_dir, 'poses_bounds.npy')
        if utils.file_exists(posefile):
            with utils.open_file(posefile, 'rb') as fp:
                poses_arr = np.load(fp)
            bounds = poses_arr[:, -2:]
        else:
            bounds = np.array([0.01, 1.])
        self.colmap_to_world_transform = np.eye(4)

        # Separate out 360 versus forward facing scenes.
        if config.forward_facing:
            # Set the projective matrix defining the NDC transformation.
            self.pixtocam_ndc = self.pixtocams.reshape(-1, 3, 3)[0]
            # Rescale according to a default bd factor.
            scale = 1. / (bounds.min() * .75)
            poses[:, :3, 3] *= scale
            self.colmap_to_world_transform = np.diag([scale] * 3 + [1])
            bounds *= scale
            # Recenter poses.
            poses, transform = camera_utils.recenter_poses(poses)
            self.colmap_to_world_transform = (
                    transform @ self.colmap_to_world_transform)
            # Forward-facing spiral render path.
            self.render_poses = camera_utils.generate_spiral_path(
                poses, bounds, n_frames=config.render_path_frames)
        else:
            # Rotate/scale poses to align ground with xy plane and fit to unit cube.
            poses, transform, _ = camera_utils.transform_poses_pca(poses)
            self.colmap_to_world_transform = transform
            if config.render_spline_keyframes is not None:
                rets = camera_utils.create_render_spline_path(config, image_names,
                                                              poses, self.exposures)
                self.spline_indices, self.render_poses, self.render_exposures = rets
            else:
                # Automatically generated inward-facing elliptical render path.
                self.render_poses = camera_utils.generate_ellipse_path(
                    poses,
                    n_frames=config.render_path_frames,
                    z_variation=config.z_variation,
                    z_phase=config.z_phase)

        # Select the split.
        all_indices = np.arange(len(image_names))
        if config.llff_use_all_images_for_training:
            train_indices = all_indices
        else:
            train_indices = all_indices % config.llffhold != 0
        split_indices = {
            utils.DataSplit.TEST: all_indices[all_indices % config.llffhold == 0],
            utils.DataSplit.TRAIN: all_indices[train_indices],
        }
        indices = split_indices[self.split]
        image_names = [image_names[i] for i in indices]
        poses = poses[indices]
        if self.split == utils.DataSplit.TRAIN:
            # load different training data on different rank
            local_indices = [i for i in range(len(image_names)) if (i + self.local_rank) % self.world_size == 0]
            image_names = [image_names[i] for i in local_indices]
            poses = poses[local_indices]

        # Scale the inverse intrinsics matrix by the image downsampling factor.
        pixtocam = pixtocam @ np.diag([factor, factor, 1.])
        self.pixtocams = pixtocam.astype(np.float32)
        self.focal = 1. / self.pixtocams[0, 0]
        self.distortion_params = distortion_params
        self.camtype = camtype

        raw_testscene = False
        if config.rawnerf_mode:
            # Load raw images and metadata.
            images, metadata, raw_testscene = raw_utils.load_raw_dataset(
                self.split,
                self.data_dir,
                image_names,
                config.exposure_percentile,
                factor)
            self.metadata = metadata

        else:
            # Load images.
            colmap_image_dir = os.path.join(self.data_dir, 'images')
            image_dir = os.path.join(self.data_dir, 'images' + image_dir_suffix)
            for d in [image_dir, colmap_image_dir]:
                if not utils.file_exists(d):
                    raise ValueError(f'Image folder {d} does not exist.')
            # Downsampled images may have different names vs images used for COLMAP,
            # so we need to map between the two sorted lists of files.
            colmap_files = sorted(utils.listdir(colmap_image_dir))
            image_files = sorted(utils.listdir(image_dir))
            colmap_to_image = dict(zip(colmap_files, image_files))
            image_paths = [os.path.join(image_dir, colmap_to_image[f])
                           for f in image_names]
            images = [utils.load_img(x) for x in tqdm(image_paths)]
            images = np.stack(images, axis=0) / 255.

            # EXIF data is usually only present in the original JPEG images.
            jpeg_paths = [os.path.join(colmap_image_dir, f) for f in image_names]
            exifs = [utils.load_exif(x) for x in jpeg_paths]
            self.exifs = exifs
            if 'ExposureTime' in exifs[0] and 'ISOSpeedRatings' in exifs[0]:
                gather_exif_value = lambda k: np.array([float(x[k]) for x in exifs])
                shutters = gather_exif_value('ExposureTime')
                isos = gather_exif_value('ISOSpeedRatings')
                self.exposures = shutters * isos / 1000.

        if raw_testscene:
            # For raw testscene, the first image sent to COLMAP has the same pose as
            # the ground truth test image. The remaining images form the training set.
            raw_testscene_poses = {
                utils.DataSplit.TEST: poses[:1],
                utils.DataSplit.TRAIN: poses[1:],
            }
            poses = raw_testscene_poses[self.split]

        self.poses = poses

        if self.exposures is not None:
            self.exposures = self.exposures[indices]
        if config.rawnerf_mode:
            for key in ['exposure_idx', 'exposure_values']:
                self.metadata[key] = self.metadata[key][indices]

        self.images = images
        self.camtoworlds = self.render_poses if config.render_path else poses
        self.height, self.width = images.shape[1:3]

from .load_nuscenes import load_waymo_meta, load_png_semantic
class WAYMO(Dataset):
    def _load_renderings(self, config):
        root_dir = self.data_dir
        self.image_list, self.poses, self.intrinsics, self.hws = load_waymo_meta(root_dir)
        poses = self.poses
        num = len(self.image_list)
        # bottom = np.zeros([num,1,4])
        # bottom[:,:,3] = 1
        # self.poses = np.concatenate([self.poses, bottom], axis=1)
        # open_fn = lambda f: utils.open_file(f, 'rb')
        # self.images = np.array([np.array(Image.open(open_fn(f))) for f in self.image_list]) / 255.

        bottom = np.zeros([num,3,1])
        self.pixtocams = np.linalg.inv(self.intrinsics)
        # self.pixtocams = np.concatenate([self.pixtocams,bottom], axis=2)
        self.camtoworlds = self.poses

        # Previous NeRF results were generated with images sorted by filename,
        # use this flag to ensure metrics are reported on the same test set.
        # inds = np.argsort(image_names)
        # image_names = [image_names[i] for i in inds]
        # poses = poses[inds]
        image_names = self.image_list

        # Load bounds if possible (only used in forward facing scenes).
        posefile = os.path.join(self.data_dir, 'poses_bounds.npy')
        if utils.file_exists(posefile):
            with utils.open_file(posefile, 'rb') as fp:
                poses_arr = np.load(fp)
            bounds = poses_arr[:, -2:]
        else:
            bounds = np.array([0.01, 1.])
        self.colmap_to_world_transform = np.eye(4)

        # Separate out 360 versus forward facing scenes.
        if config.forward_facing:
            # Set the projective matrix defining the NDC transformation.
            self.pixtocam_ndc = self.pixtocams.reshape(-1, 3, 3)[0]
            # Rescale according to a default bd factor.
            scale = 1. / (bounds.min() * .75)
            poses[:, :3, 3] *= scale
            self.colmap_to_world_transform = np.diag([scale] * 3 + [1])
            bounds *= scale
            # Recenter poses.
            poses, transform = camera_utils.recenter_poses(poses)
            self.colmap_to_world_transform = (
                    transform @ self.colmap_to_world_transform)
            # Forward-facing spiral render path.
            self.render_poses = camera_utils.generate_spiral_path(
                poses, bounds, n_frames=config.render_path_frames)
        else:
            # Rotate/scale poses to align ground with xy plane and fit to unit cube.
            poses, transform, scale_factor = camera_utils.transform_poses_pca(poses) # scale=1/max_d
            ## auto scale near and far
            self.scale_factor = scale_factor
            self.near = 2*scale_factor
            self.far = 10000*scale_factor

            if self.local_rank == 0: print('The scale factor of the scene is {}'.format(1/scale_factor))
            self.colmap_to_world_transform = transform
            if config.render_spline_keyframes is not None:
                rets = camera_utils.create_render_spline_path(config, image_names,
                                                              poses, self.exposures)
                self.spline_indices, self.render_poses, self.render_exposures = rets
            else:
                # Automatically generated inward-facing elliptical render path.
                self.render_poses = camera_utils.generate_ellipse_path(
                    poses,
                    n_frames=config.render_path_frames,
                    z_variation=config.z_variation,
                    z_phase=config.z_phase)

        # Select the split.
        all_indices = np.arange(len(image_names))
        if config.llff_use_all_images_for_training:
            train_indices = all_indices
        else:
            train_indices = all_indices % config.llffhold != 0
        split_indices = {
            utils.DataSplit.TEST: all_indices[all_indices % config.llffhold == 0],
            utils.DataSplit.TRAIN: all_indices[train_indices],
        }
        indices = split_indices[self.split]

        ### for debug
        # indices = indices[:3]
        if config.debug:
            indices = indices[:3]

        num_images = len(image_names)
        self.num_poses = len(indices) # used for pose refine

        image_names = [image_names[i] for i in indices]
        depth_names = [img.replace('images', 'depth') for img in image_names]
        semantic_names = [img.replace('images', 'labels') for img in image_names]
        mask_names = [img.replace('images', 'mask') for img in image_names]
        def load_mask(mask_path, num_images=250):
            idx = int(mask_path[-8:-4])
            num = num_images//5
            if os.path.exists(mask_path):
                mask = imageio.imread(mask_path)
                mask = np.asarray(mask).copy()
                # import pdb; pdb.set_trace()
                mask[mask > 0] = 1
            else:
                mask = np.zeros([1280, 1920])
                if idx >= 3 * num:
                    mask[886:] = 1
            return mask

        poses = poses[indices]
        self.pixtocams = self.pixtocams[indices]
        if self.split == utils.DataSplit.TRAIN:
            # load different training data on different rank
            local_indices = [i for i in range(len(image_names)) if (i + self.local_rank) % self.world_size == 0]
            self.local2global_idx = np.array(local_indices)
            image_names = [image_names[i] for i in local_indices]
            depth_names = [depth_names[i] for i in local_indices]
            semantic_names = [semantic_names[i] for i in local_indices]
            poses = poses[local_indices]
            mask_names = [mask_names[i] for i in local_indices]
            self.pixtocams = self.pixtocams[local_indices]

        raw_testscene = False
        if config.rawnerf_mode:
            # Load raw images and metadata.
            images, metadata, raw_testscene = raw_utils.load_raw_dataset(
                self.split,
                self.data_dir,
                image_names,
                config.exposure_percentile,
                factor)
            self.metadata = metadata

        else:

            image_paths = image_names
            if self.local_rank==0: print('|------load image-----|')
            images = [utils.load_img(x) for x in tqdm(image_paths)]
            images = np.stack(images, axis=0) / 255.
            depths = [cv2.imread(f, -1) / 256. * scale_factor for f in tqdm(depth_names)]
            depths = np.stack(depths, axis=0).astype(np.float32)
            if self.local_rank==0: print('|------load semantic-----|')
            semantics = load_png_semantic(semantic_names)
            if self.local_rank==0: print('|------load mask-----|')
            masks = [load_mask(f, num_images=num_images) for f in tqdm(mask_names)]
            masks = np.stack(masks, axis=0)



        if raw_testscene:
            # For raw testscene, the first image sent to COLMAP has the same pose as
            # the ground truth test image. The remaining images form the training set.
            raw_testscene_poses = {
                utils.DataSplit.TEST: poses[:1],
                utils.DataSplit.TRAIN: poses[1:],
            }
            poses = raw_testscene_poses[self.split]

        self.poses = poses

        if self.exposures is not None:
            self.exposures = self.exposures[indices]
        if config.rawnerf_mode:
            for key in ['exposure_idx', 'exposure_values']:
                self.metadata[key] = self.metadata[key][indices]

        self.images = images
        self.camtoworlds = self.render_poses if config.render_path else poses
        self.height, self.width = images.shape[1:3]
        self.depths = depths
        self.semantics = semantics
        self.masks = masks


class WAYMO_RENDER(Dataset):
    def _load_renderings(self, config):
        root_dir = self.data_dir
        self.image_list, self.poses, self.intrinsics, self.hws = load_waymo_meta(root_dir)
        poses = self.poses
        num = len(self.image_list)
        # self.pixtocams = np.linalg.inv(self.intrinsics)
        # # self.pixtocams = np.concatenate([self.pixtocams,bottom], axis=2)
        # self.camtoworlds = self.poses
        #
        # image_names = self.image_list

        # Load bounds if possible (only used in forward facing scenes).
        posefile = os.path.join(self.data_dir, 'poses_bounds.npy')
        if utils.file_exists(posefile):
            with utils.open_file(posefile, 'rb') as fp:
                poses_arr = np.load(fp)
            bounds = poses_arr[:, -2:]
        else:
            bounds = np.array([0.01, 1.])
        self.colmap_to_world_transform = np.eye(4)

        # Rotate/scale poses to align ground with xy plane and fit to unit cube.
        poses, transform, scale_factor = camera_utils.transform_poses_pca(poses)  # scale=1/max_d
        ## auto scale near and far
        self.scale_factor = scale_factor
        self.near = 2 * scale_factor
        self.far = 10000 * scale_factor

        if self.local_rank == 0: print('The scale factor of the scene is {}'.format(1 / scale_factor))
        self.colmap_to_world_transform = transform
        # if config.render_spline_keyframes is not None:
        #     rets = camera_utils.create_render_spline_path(config, image_names,
        #                                                   poses, self.exposures)
        #     self.spline_indices, self.render_poses, self.render_exposures = rets
        # else:
        #     # Automatically generated inward-facing elliptical render path.
        #     self.render_poses = camera_utils.generate_ellipse_path(
        #         poses,
        #         n_frames=config.render_path_frames,
        #         z_variation=config.z_variation,
        #         z_phase=config.z_phase)
        ##########################################
        #        render poses generation         #
        ##########################################
        bottom = np.zeros([num, 1, 4])
        poses_bt = np.concatenate([poses, bottom], axis=1)
        poses_bt[:, 3, 3] = 1
        poses = poses_bt

        total_poses_num = poses.shape[0]
        forward_poses = poses[:total_poses_num // 5, ...]

        K_forward_cam = self.intrinsics[0]
        _temp = self.intrinsics[-1]
        HW_forward_cam = np.array([1280, 1920])
        # _temp[1,2] = 245
        K_side_cam = _temp
        HW_side_cam = np.array([886, 1920])

        render_poses = []
        Ks = []
        hw = []

        dx, dy, dz = np.array([.5, .5, .25])*scale_factor
        dx_theta, dy_theta = 2.5, 5
        import random
        # random.seed(seed_val)  # The same random for all parts!!!!!!!
        for idx in range(config.RENDER_N):
            ##############################################################
            # random_idx = random.randint(0, total_poses_num // 5 - 1)     # Only forward camera.
            ##############################################################
            # random_idx = random.randint(0, total_poses_num * 3 // 5 - 1) # Only forward 3 cameras.
            ##############################################################

            def random_v2():
                '''
                disturbance is samll, interplote from adjacent frame
                :return:
                '''
                frame_num = total_poses_num // 5
                frame_offset = 5
                if config.only_side_cam:
                    random_idx = random.randint(3, 4) * frame_num + random.randint(frame_offset,
                                                                                   frame_num - frame_offset)

                elif config.only_front_cam:
                    ### !!! for only front means the first 3 camera for this version (only front 3 camera version)
                    random_idx = random.randint(0, 2) * frame_num + random.randint(frame_offset,
                                                                                   frame_num - frame_offset)
                    ##(only front 1 camera version)
                    # random_idx = 0 * frame_num + random.randint(frame_offset, frame_num - frame_offset)
                else:
                    random_idx = random.randint(0, 4) * frame_num + random.randint(frame_offset,
                                                                                   frame_num - frame_offset)

                frame_id = random_idx % frame_num  ## frame id, should be [offset,num-offset]
                id_part = random_idx // frame_num

                if random_idx > total_poses_num * 3 // 5 - 1:  # Side cameras.
                    ano_set = [random_idx, random_idx + 1, random_idx - 1]
                    ano_random_idx = ano_set[random.randint(0, 2)]  # Only interpolate in side 2 cameras.
                    pose_0, pose_1 = poses[random_idx, ...].copy(), poses[ano_random_idx, ...].copy()
                    pose = interpolate_two_pose(pose_0, pose_1, ratio=random.random(), fix_trans=True)
                    Ks.append(K_side_cam)
                    hw.append(HW_side_cam)
                else:
                    # if config.only_front_cam:  # Not to interpolate pose.  (only front 1 camera version)
                    if False:  # !!! for only front means the first 3 camera for this version (only front 3 camera version)
                        pose = poses[random_idx, ...].copy()
                        pose = add_noise_to_pose(pose, dx, dy, dz, dx_theta, dy_theta)
                    else:
                        # this method lead to tend to center the camera
                        # ano_set = [0*frame_num+frame_id,0*frame_num+frame_id-1,0*frame_num+frame_id+1,
                        #            1*frame_num+frame_id,1*frame_num+frame_id-1,1*frame_num+frame_id+1,
                        #            2*frame_num+frame_id,2*frame_num+frame_id-1,2*frame_num+frame_id+1,]

                        part_ano = random.randint(1, 2) if id_part == 0 else 0
                        ano_set = [random_idx + 1, random_idx, random_idx - 1,
                                   part_ano * frame_num + frame_id, part_ano * frame_num + frame_id - 1,
                                   part_ano * frame_num + frame_id + 1, ]
                        ano_random_idx = ano_set[random.randint(0, 5)]
                        pose_0, pose_1 = poses[random_idx, ...].copy(), poses[ano_random_idx, ...].copy()
                        pose_0, pose_1 = \
                            add_noise_to_pose(pose_0, dx, dy, dz, dx_theta, dy_theta), \
                            add_noise_to_pose(pose_1, dx, dy, dz, dx_theta, dy_theta)
                        pose = interpolate_two_pose(pose_0, pose_1, ratio=random.random(), fix_trans=False)
                    Ks.append(K_forward_cam)
                    hw.append(HW_forward_cam)
                render_poses.append(pose)
            random_v2()
        render_poses = np.stack(render_poses, axis=0)
        render_poses_sd = render_poses+0
        render_poses_sd[:,:3,3] = render_poses_sd[:,:3,3]/scale_factor
        Ks = np.stack(Ks, axis=0)
        hws = np.stack(hw, axis=0)
        self.poses = render_poses
        self.camtoworlds = render_poses
        self.pixtocams = np.linalg.inv(Ks)
        self.hws = hws
        self.intrinsic_fwd = K_forward_cam
        self.render_poses_sd = render_poses_sd

        def render2raw():
            transform_uni = np.diag(np.array([1/scale_factor] * 3 + [1])) @ transform
            raw_poses = render_poses+0
            raw_poses[:,:3,3] = raw_poses[:,:3,3]/scale_factor
            raw_poses = np.linalg.inv(transform_uni[None,...]) @ raw_poses
            c2w = np.load(os.path.join(config.data_dir, 'c2w.npy'))
            start_raw = c2w[0,0]
            raw_poses = np.concatenate(
                [raw_poses[:, :, 0:1], -raw_poses[:, :, 1:2], -raw_poses[:, :, 2:3], raw_poses[:, :, 3:4]], -1)
            raw_poses = start_raw[None,...] @ raw_poses
            self.raw_poses = raw_poses

        render2raw()



        # self.camtoworlds = self.render_poses if config.render_path else poses
        self.height, self.width = [1280, 1920]
        # self.depths = depths
        # self.semantics = semantics
        # self.masks = masks

        # # Select the split.
        # all_indices = np.arange(config.RENDER_N)
        # if config.llff_use_all_images_for_training:
        #     train_indices = all_indices
        # else:
        #     train_indices = all_indices % config.llffhold != 0
        # split_indices = {
        #     utils.DataSplit.TEST: all_indices[all_indices % config.llffhold == 0],
        #     utils.DataSplit.TRAIN: all_indices[train_indices],
        # }
        # indices = split_indices[self.split]
        #
        # ### for debug
        # # indices = indices[:3]
        # if config.debug:
        #     indices = indices[:3]
        #
        # num_images = len(image_names)
        # self.num_poses = len(indices) # used for pose refine
        #
        # image_names = [image_names[i] for i in indices]
        # depth_names = [img.replace('images', 'depth') for img in image_names]
        # semantic_names = [img.replace('images', 'labels') for img in image_names]
        # mask_names = [img.replace('images', 'mask') for img in image_names]
        # def load_mask(mask_path, num_images=250):
        #     idx = int(mask_path[-8:-4])
        #     num = num_images//5
        #     if os.path.exists(mask_path):
        #         mask = imageio.imread(mask_path)
        #         mask = np.asarray(mask).copy()
        #         # import pdb; pdb.set_trace()
        #         mask[mask > 0] = 1
        #     else:
        #         mask = np.zeros([1280, 1920])
        #         if idx >= 3 * num:
        #             mask[886:] = 1
        #     return mask
        #
        # poses = poses[indices]
        # self.pixtocams = self.pixtocams[indices]
        # if self.split == utils.DataSplit.TRAIN:
        #     # load different training data on different rank
        #     local_indices = [i for i in range(len(image_names)) if (i + self.local_rank) % self.world_size == 0]
        #     self.local2global_idx = np.array(local_indices)
        #     image_names = [image_names[i] for i in local_indices]
        #     depth_names = [depth_names[i] for i in local_indices]
        #     semantic_names = [semantic_names[i] for i in local_indices]
        #     poses = poses[local_indices]
        #     mask_names = [mask_names[i] for i in local_indices]
        #     self.pixtocams = self.pixtocams[local_indices]

        # raw_testscene = False
        # if config.rawnerf_mode:
        #     # Load raw images and metadata.
        #     images, metadata, raw_testscene = raw_utils.load_raw_dataset(
        #         self.split,
        #         self.data_dir,
        #         image_names,
        #         config.exposure_percentile,
        #         factor)
        #     self.metadata = metadata
        #
        # else:
        #
        #     image_paths = image_names
        #     if self.local_rank==0: print('|------load image-----|')
        #     images = [utils.load_img(x) for x in tqdm(image_paths)]
        #     images = np.stack(images, axis=0) / 255.
        #     depths = [cv2.imread(f, -1) / 256. * scale_factor for f in tqdm(depth_names)]
        #     depths = np.stack(depths, axis=0).astype(np.float32)
        #     if self.local_rank==0: print('|------load semantic-----|')
        #     semantics = load_png_semantic(semantic_names)
        #     if self.local_rank==0: print('|------load mask-----|')
        #     masks = [load_mask(f, num_images=num_images) for f in tqdm(mask_names)]
        #     masks = np.stack(masks, axis=0)


        #
        # if raw_testscene:
        #     # For raw testscene, the first image sent to COLMAP has the same pose as
        #     # the ground truth test image. The remaining images form the training set.
        #     raw_testscene_poses = {
        #         utils.DataSplit.TEST: poses[:1],
        #         utils.DataSplit.TRAIN: poses[1:],
        #     }
        #     poses = raw_testscene_poses[self.split]

        # self.poses = poses
        #
        # if self.exposures is not None:
        #     self.exposures = self.exposures[indices]
        # if config.rawnerf_mode:
        #     for key in ['exposure_idx', 'exposure_values']:
        #         self.metadata[key] = self.metadata[key][indices]

        # self.images = imagescc


class WAYMO_DEMO(Dataset):
    def _load_renderings(self, config):
        root_dir = self.data_dir
        self.image_list, self.poses, self.intrinsics, self.hws = load_waymo_meta(root_dir)
        poses = self.poses
        num = len(self.image_list)

        posefile = os.path.join(self.data_dir, 'poses_bounds.npy')
        if utils.file_exists(posefile):
            with utils.open_file(posefile, 'rb') as fp:
                poses_arr = np.load(fp)
            bounds = poses_arr[:, -2:]
        else:
            bounds = np.array([0.01, 1.])
        self.colmap_to_world_transform = np.eye(4)

        # Rotate/scale poses to align ground with xy plane and fit to unit cube.
        poses, transform, scale_factor = camera_utils.transform_poses_pca(poses)  # scale=1/max_d
        ## auto scale near and far
        self.scale_factor = scale_factor
        self.near = 2 * scale_factor
        self.far = 10000 * scale_factor

        if self.local_rank == 0: print('The scale factor of the scene is {}'.format(1 / scale_factor))
        self.colmap_to_world_transform = transform
        ##########################################
        #        render poses generation         #
        ##########################################
        bottom = np.zeros([num, 1, 4])
        poses_bt = np.concatenate([poses, bottom], axis=1)
        poses_bt[:, 3, 3] = 1
        poses = poses_bt

        total_poses_num = poses.shape[0]
        forward_poses = poses[:total_poses_num // 5, ...]

        K_forward_cam = self.intrinsics[0]
        _temp = self.intrinsics[-1]
        HW_forward_cam = np.array([1280, 1920])
        # _temp[1,2] = 245
        K_side_cam = _temp
        HW_side_cam = np.array([886, 1920])

        render_poses = []
        Ks = []
        hw = []

        dx, dy, dz = np.array([.5, .5, .25])*scale_factor
        dx_theta, dy_theta = 2.5, 5
        import random
        # random.seed(seed_val)  # The same random for all parts!!!!!!!
        for idx in range(config.RENDER_N):
            ##############################################################
            # random_idx = random.randint(0, total_poses_num // 5 - 1)     # Only forward camera.
            ##############################################################
            # random_idx = random.randint(0, total_poses_num * 3 // 5 - 1) # Only forward 3 cameras.
            ##############################################################
            frame_num = total_poses_num // 5
            frame_offset = 5
            start_frame = frame_offset
            def random_v2():
                '''
                disturbance is samll, interplote from adjacent frame
                :return:
                '''
                random_idx = start_frame+idx
                pose = poses[random_idx, ...].copy()
                Ks.append(K_forward_cam)
                hw.append(HW_forward_cam)
                render_poses.append(pose)
            random_v2()
        render_poses = np.stack(render_poses, axis=0)
        render_poses_sd = render_poses+0
        render_poses_sd[:,:3,3] = render_poses_sd[:,:3,3]/scale_factor
        Ks = np.stack(Ks, axis=0)
        hws = np.stack(hw, axis=0)
        self.poses = render_poses
        self.camtoworlds = render_poses
        self.pixtocams = np.linalg.inv(Ks)
        self.hws = hws
        self.intrinsic_fwd = K_forward_cam
        self.render_poses_sd = render_poses_sd

        def render2raw():
            transform_uni = np.diag(np.array([1/scale_factor] * 3 + [1])) @ transform
            raw_poses = render_poses+0
            raw_poses[:,:3,3] = raw_poses[:,:3,3]/scale_factor
            raw_poses = np.linalg.inv(transform_uni[None,...]) @ raw_poses
            c2w = np.load(os.path.join(config.data_dir, 'c2w.npy'))
            start_raw = c2w[0,0]
            raw_poses = np.concatenate(
                [raw_poses[:, :, 0:1], -raw_poses[:, :, 1:2], -raw_poses[:, :, 2:3], raw_poses[:, :, 3:4]], -1)
            raw_poses = start_raw[None,...] @ raw_poses
            self.raw_poses = raw_poses

        render2raw()



        # self.camtoworlds = self.render_poses if config.render_path else poses
        self.height, self.width = [1280, 1920]
        # self.depths = depths
        # self.semantics = semantics
        # self.masks = masks

        # # Select the split.
        # all_indices = np.arange(config.RENDER_N)
        # if config.llff_use_all_images_for_training:
        #     train_indices = all_indices
        # else:
        #     train_indices = all_indices % config.llffhold != 0
        # split_indices = {
        #     utils.DataSplit.TEST: all_indices[all_indices % config.llffhold == 0],
        #     utils.DataSplit.TRAIN: all_indices[train_indices],
        # }
        # indices = split_indices[self.split]
        #
        # ### for debug
        # # indices = indices[:3]
        # if config.debug:
        #     indices = indices[:3]
        #
        # num_images = len(image_names)
        # self.num_poses = len(indices) # used for pose refine
        #
        # image_names = [image_names[i] for i in indices]
        # depth_names = [img.replace('images', 'depth') for img in image_names]
        # semantic_names = [img.replace('images', 'labels') for img in image_names]
        # mask_names = [img.replace('images', 'mask') for img in image_names]
        # def load_mask(mask_path, num_images=250):
        #     idx = int(mask_path[-8:-4])
        #     num = num_images//5
        #     if os.path.exists(mask_path):
        #         mask = imageio.imread(mask_path)
        #         mask = np.asarray(mask).copy()
        #         # import pdb; pdb.set_trace()
        #         mask[mask > 0] = 1
        #     else:
        #         mask = np.zeros([1280, 1920])
        #         if idx >= 3 * num:
        #             mask[886:] = 1
        #     return mask
        #
        # poses = poses[indices]
        # self.pixtocams = self.pixtocams[indices]
        # if self.split == utils.DataSplit.TRAIN:
        #     # load different training data on different rank
        #     local_indices = [i for i in range(len(image_names)) if (i + self.local_rank) % self.world_size == 0]
        #     self.local2global_idx = np.array(local_indices)
        #     image_names = [image_names[i] for i in local_indices]
        #     depth_names = [depth_names[i] for i in local_indices]
        #     semantic_names = [semantic_names[i] for i in local_indices]
        #     poses = poses[local_indices]
        #     mask_names = [mask_names[i] for i in local_indices]
        #     self.pixtocams = self.pixtocams[local_indices]

        # raw_testscene = False
        # if config.rawnerf_mode:
        #     # Load raw images and metadata.
        #     images, metadata, raw_testscene = raw_utils.load_raw_dataset(
        #         self.split,
        #         self.data_dir,
        #         image_names,
        #         config.exposure_percentile,
        #         factor)
        #     self.metadata = metadata
        #
        # else:
        #
        #     image_paths = image_names
        #     if self.local_rank==0: print('|------load image-----|')
        #     images = [utils.load_img(x) for x in tqdm(image_paths)]
        #     images = np.stack(images, axis=0) / 255.
        #     depths = [cv2.imread(f, -1) / 256. * scale_factor for f in tqdm(depth_names)]
        #     depths = np.stack(depths, axis=0).astype(np.float32)
        #     if self.local_rank==0: print('|------load semantic-----|')
        #     semantics = load_png_semantic(semantic_names)
        #     if self.local_rank==0: print('|------load mask-----|')
        #     masks = [load_mask(f, num_images=num_images) for f in tqdm(mask_names)]
        #     masks = np.stack(masks, axis=0)


        #
        # if raw_testscene:
        #     # For raw testscene, the first image sent to COLMAP has the same pose as
        #     # the ground truth test image. The remaining images form the training set.
        #     raw_testscene_poses = {
        #         utils.DataSplit.TEST: poses[:1],
        #         utils.DataSplit.TRAIN: poses[1:],
        #     }
        #     poses = raw_testscene_poses[self.split]

        # self.poses = poses
        #
        # if self.exposures is not None:
        #     self.exposures = self.exposures[indices]
        # if config.rawnerf_mode:
        #     for key in ['exposure_idx', 'exposure_values']:
        #         self.metadata[key] = self.metadata[key][indices]

        # self.images = imagescc



class NUSCENES_RENDER(Dataset):
    def _load_renderings(self, config):
        root_dir = self.data_dir
        self.image_list, self.poses, self.intrinsics, self.hws = load_waymo_meta(root_dir)
        poses = self.poses
        num = len(self.image_list)
        # self.pixtocams = np.linalg.inv(self.intrinsics)
        # # self.pixtocams = np.concatenate([self.pixtocams,bottom], axis=2)
        # self.camtoworlds = self.poses
        #
        # image_names = self.image_list

        # Load bounds if possible (only used in forward facing scenes).
        posefile = os.path.join(self.data_dir, 'poses_bounds.npy')
        if utils.file_exists(posefile):
            with utils.open_file(posefile, 'rb') as fp:
                poses_arr = np.load(fp)
            bounds = poses_arr[:, -2:]
        else:
            bounds = np.array([0.01, 1.])
        self.colmap_to_world_transform = np.eye(4)

        # Rotate/scale poses to align ground with xy plane and fit to unit cube.
        poses, transform, scale_factor = camera_utils.transform_poses_pca(poses)  # scale=1/max_d
        ## auto scale near and far
        self.scale_factor = scale_factor
        self.near = 2 * scale_factor
        self.far = 10000 * scale_factor

        if self.local_rank == 0: print('The scale factor of the scene is {}'.format(1 / scale_factor))
        self.colmap_to_world_transform = transform
        # if config.render_spline_keyframes is not None:
        #     rets = camera_utils.create_render_spline_path(config, image_names,
        #                                                   poses, self.exposures)
        #     self.spline_indices, self.render_poses, self.render_exposures = rets
        # else:
        #     # Automatically generated inward-facing elliptical render path.
        #     self.render_poses = camera_utils.generate_ellipse_path(
        #         poses,
        #         n_frames=config.render_path_frames,
        #         z_variation=config.z_variation,
        #         z_phase=config.z_phase)
        ##########################################
        #        render poses generation         #
        ##########################################
        bottom = np.zeros([num, 1, 4])
        poses_bt = np.concatenate([poses, bottom], axis=1)
        poses_bt[:, 3, 3] = 1
        poses = poses_bt

        total_poses_num = poses.shape[0]
        chunk = total_poses_num//6
        forward_poses = poses[2*chunk:3*chunk, ...]

        K_forward_cam = CAM_FRONT_['cam_intrinsic']
        _temp = self.intrinsics[-1]
        HW_forward_cam = np.array([900, 1600])

        render_poses = []
        Ks = []
        hw = []

        dx, dy, dz = np.array([.5, .5, .25])*scale_factor
        dx_theta, dy_theta = 2.5, 5
        import random
        # random.seed(seed_val)  # The same random for all parts!!!!!!!
        for idx in range(config.RENDER_N):
            ##############################################################
            # random_idx = random.randint(0, total_poses_num // 5 - 1)     # Only forward camera.
            ##############################################################
            # random_idx = random.randint(0, total_poses_num * 3 // 5 - 1) # Only forward 3 cameras.
            ##############################################################
            def random_v2():
                front_cam_start = total_poses_num // 2
                Ks = []
                hw = []
                fron_cam_num = total_poses_num // 6
                frame_offset = 10
                random_idx = random.randint(front_cam_start + frame_offset,
                                            front_cam_start + fron_cam_num - frame_offset)

                pose = poses[random_idx, ...].copy()
                pose = add_noise_to_pose(pose, dx, dy, dz, dx_theta, dy_theta)
                render_poses.append(pose)
                Ks.append(K_forward_cam)
                hw.append(HW_forward_cam)

                for cam in sorted(list(sensor_dict.keys())):
                    if cam == 'CAM_FRONT' or cam == 'LIDAR':
                        continue
                    front2world = pose
                    sensor2front = get_a2b(sensor_dict[cam], CAM_FRONT_)
                    sensor2world = sensor2front@front2world
                    render_poses.append(sensor2world)
                    Ks.append(sensor_dict[cam]['intrinsic'])
                    hw.append(HW_forward_cam)

            random_v2()
        render_poses = np.stack(render_poses, axis=0)
        render_poses_sd = render_poses+0
        render_poses_sd[:,:3,3] = render_poses_sd[:,:3,3]/scale_factor
        Ks = np.stack(Ks, axis=0)
        hws = np.stack(hw, axis=0)
        self.poses = render_poses
        self.camtoworlds = render_poses
        self.pixtocams = np.linalg.inv(Ks)
        self.hws = hws
        self.intrinsic_fwd = K_forward_cam
        self.render_poses_sd = render_poses_sd

        def render2raw():
            transform_uni = np.diag(np.array([1/scale_factor] * 3 + [1])) @ transform
            raw_poses = render_poses+0
            raw_poses[:,:3,3] = raw_poses[:,:3,3]/scale_factor
            raw_poses = np.linalg.inv(transform_uni[None,...]) @ raw_poses
            c2w = np.load(os.path.join(config.data_dir, 'c2w.npy'))
            start_raw = c2w[0,0]
            raw_poses = np.concatenate(
                [raw_poses[:, :, 0:1], -raw_poses[:, :, 1:2], -raw_poses[:, :, 2:3], raw_poses[:, :, 3:4]], -1)
            raw_poses = start_raw[None,...] @ raw_poses
            self.raw_poses = raw_poses

        render2raw()



class NUSCENES(Dataset):
    def _load_renderings(self, config):
        root_dir = self.data_dir
        self.image_list, self.poses, self.intrinsics, self.hws = load_waymo_meta(root_dir)
        poses = self.poses
        num = len(self.image_list)
        # import pdb;pdb.set_trace()
        # bottom = np.zeros([num,3,1])
        self.pixtocams = np.linalg.inv(self.intrinsics)

        self.camtoworlds = self.poses

        # Previous NeRF results were generated with images sorted by filename,
        # use this flag to ensure metrics are reported on the same test set.
        # inds = np.argsort(image_names)
        # image_names = [image_names[i] for i in inds]
        # poses = poses[inds]
        image_names = self.image_list

        # Load bounds if possible (only used in forward facing scenes).
        posefile = os.path.join(self.data_dir, 'poses_bounds.npy')
        if utils.file_exists(posefile):
            with utils.open_file(posefile, 'rb') as fp:
                poses_arr = np.load(fp)
            bounds = poses_arr[:, -2:]
        else:
            bounds = np.array([0.01, 1.])
        self.colmap_to_world_transform = np.eye(4)

        # Separate out 360 versus forward facing scenes.
        if config.forward_facing:
            # Set the projective matrix defining the NDC transformation.
            self.pixtocam_ndc = self.pixtocams.reshape(-1, 3, 3)[0]
            # Rescale according to a default bd factor.
            scale = 1. / (bounds.min() * .75)
            poses[:, :3, 3] *= scale
            self.colmap_to_world_transform = np.diag([scale] * 3 + [1])
            bounds *= scale
            # Recenter poses.
            poses, transform = camera_utils.recenter_poses(poses)
            self.colmap_to_world_transform = (
                    transform @ self.colmap_to_world_transform)
            # Forward-facing spiral render path.
            self.render_poses = camera_utils.generate_spiral_path(
                poses, bounds, n_frames=config.render_path_frames)
        else:
            # Rotate/scale poses to align ground with xy plane and fit to unit cube.
            poses, transform, scale_factor = camera_utils.transform_poses_pca(poses) # scale=1/max_d
            self.scale_factor = scale_factor
            self.near = 2*scale_factor
            self.far = 10000*scale_factor
            self.colmap_to_world_transform = transform
            if config.render_spline_keyframes is not None:
                rets = camera_utils.create_render_spline_path(config, image_names,
                                                              poses, self.exposures)
                self.spline_indices, self.render_poses, self.render_exposures = rets
            else:
                # Automatically generated inward-facing elliptical render path.
                self.render_poses = camera_utils.generate_ellipse_path(
                    poses,
                    n_frames=config.render_path_frames,
                    z_variation=config.z_variation,
                    z_phase=config.z_phase)

        # Select the split.
        all_indices = np.arange(len(image_names))
        if config.llff_use_all_images_for_training:
            train_indices = all_indices
        else:
            train_indices = all_indices % config.llffhold != 0
        split_indices = {
            utils.DataSplit.TEST: all_indices[all_indices % config.llffhold == 0],
            utils.DataSplit.TRAIN: all_indices[train_indices],
        }
        indices = split_indices[self.split]

        ### for debug

        num_images = len(image_names)
        self.num_poses = len(indices) # used for pose refine

        image_names = [image_names[i] for i in indices]
        depth_names = [img.replace('images', 'depth') for img in image_names]
        semantic_names = [img.replace('images', 'labels') for img in image_names]
        mask_names = [img.replace('images', 'mask') for img in image_names]
        def load_mask(datadir, num_images=250,segmentation = None,indices= None):
            H, W = 900, 1600
            def load_moving_mask(datadir,indices= None,segmentation= None , num_image=num_images):
                mask_path = os.path.join(datadir,'mask')
                im_masks = []
                # if segmenation is not None:
                count = 0
                num = num_image//6 # ori 5
                # import pdb;pdb.set_trace()
                masks = sorted(os.listdir(mask_path))
                if indices is not None:
                    masks = [masks[i] for i in indices]
                for frame_mask in masks:
                    tmp_mask = np.ones((H, W))
                    with open(os.path.join(datadir,'mask',frame_mask),'r') as f:
                        mask_infos = f.readlines()
                        mask_infos = np.array([mask_info_.split()[1:] for mask_info_ in mask_infos]).astype(np.int16)

                    for mask in mask_infos:
                        # print(self.local_rank,':',segmentation.shape,' ',len(masks))
                        # print(self.local_rank,':', len(indices), ' ',indices[0])
                        tmp_seg = segmentation[count][mask[0]:mask[2],mask[1]:mask[3]]>=11
                        #only mask the related category,i.e. person, vehicle and so on.
                        tmp_mask[mask[0]:mask[2],mask[1]:mask[3]] = tmp_seg==0
                    if indices[count] < num:
                        tmp_mask[800:,:] = 0
                    count +=1

                    im_masks.append(tmp_mask)
                return im_masks
            img_masks = load_moving_mask(datadir,segmentation=segmentation,indices=indices)
            mask = np.stack(img_masks,axis=0) # N,H,W

            return mask

        poses = poses[indices]
        self.pixtocams = self.pixtocams[indices]
        if self.split == utils.DataSplit.TRAIN:
            # load different training data on different rank
            local_indices = [i for i in range(len(image_names)) if (i + self.local_rank) % self.world_size == 0]
            self.local2global_idx = np.array(local_indices)
            image_names = [image_names[i] for i in local_indices]
            depth_names = [depth_names[i] for i in local_indices]
            semantic_names = [semantic_names[i] for i in local_indices]
            poses = poses[local_indices]
            mask_names = [mask_names[i] for i in local_indices]
            self.pixtocams = self.pixtocams[local_indices]
        else:
            local_indices = indices

        raw_testscene = False
        if config.rawnerf_mode:
            # Load raw images and metadata.
            images, metadata, raw_testscene = raw_utils.load_raw_dataset(
                self.split,
                self.data_dir,
                image_names,
                config.exposure_percentile,
                factor)
            self.metadata = metadata

        else:

            image_paths = image_names
            if self.local_rank==0: print('|------load image-----|')
            images = [utils.load_img(x) for x in tqdm(image_paths)]
            images = np.stack(images, axis=0) / 255.
            print('|------load depth-----|')
            depths = [cv2.imread(f, -1) / 256. * scale_factor for f in tqdm(depth_names)]
            depths = np.stack(depths, axis=0).astype(np.float32)
            if self.local_rank==0: print('|------load semantic-----|')
            semantics = load_png_semantic(semantic_names)
            if self.local_rank==0: print('|------load mask-----|')
            # masks = [load_mask(f, num_images=num_images) for f in tqdm(mask_names)]
            # masks = np.stack(masks, axis=0)
            masks = load_mask(self.data_dir, num_images=num_images,
                              segmentation=semantics,indices=local_indices)
            # import pdb;pdb.set_trace()


        if raw_testscene:
            # For raw testscene, the first image sent to COLMAP has the same pose as
            # the ground truth test image. The remaining images form the training set.
            raw_testscene_poses = {
                utils.DataSplit.TEST: poses[:1],
                utils.DataSplit.TRAIN: poses[1:],
            }
            poses = raw_testscene_poses[self.split]

        self.poses = poses

        if self.exposures is not None:
            self.exposures = self.exposures[indices]
        if config.rawnerf_mode:
            for key in ['exposure_idx', 'exposure_values']:
                self.metadata[key] = self.metadata[key][indices]

        self.images = images
        self.camtoworlds = self.render_poses if config.render_path else poses
        self.height, self.width = images.shape[1:3]
        self.depths = depths
        self.semantics = semantics
        self.masks = masks




class TanksAndTemplesNerfPP(Dataset):
    """Subset of Tanks and Temples Dataset as processed by NeRF++."""

    def _load_renderings(self, config):
        """Load images from disk."""
        if config.render_path:
            split_str = 'camera_path'
        else:
            split_str = self.split.value

        basedir = os.path.join(self.data_dir, split_str)

        def load_files(dirname, load_fn, shape=None):
            files = [
                os.path.join(basedir, dirname, f)
                for f in sorted(utils.listdir(os.path.join(basedir, dirname)))
            ]
            mats = np.array([load_fn(utils.open_file(f, 'rb')) for f in files])
            if shape is not None:
                mats = mats.reshape(mats.shape[:1] + shape)
            return mats

        poses = load_files('pose', np.loadtxt, (4, 4))
        # Flip Y and Z axes to get correct coordinate frame.
        poses = np.matmul(poses, np.diag(np.array([1, -1, -1, 1])))

        # For now, ignore all but the first focal length in intrinsics
        intrinsics = load_files('intrinsics', np.loadtxt, (4, 4))

        if not config.render_path:
            images = load_files('rgb', lambda f: np.array(Image.open(f))) / 255.
            self.images = images
            self.height, self.width = self.images.shape[1:3]

        else:
            # Hack to grab the image resolution from a test image
            d = os.path.join(self.data_dir, 'test', 'rgb')
            f = os.path.join(d, sorted(utils.listdir(d))[0])
            shape = utils.load_img(f).shape
            self.height, self.width = shape[:2]
            self.images = None

        self.camtoworlds = poses
        self.focal = intrinsics[0, 0, 0]
        self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                   self.height)


class TanksAndTemplesFVS(Dataset):
    """Subset of Tanks and Temples Dataset as processed by Free View Synthesis."""

    def _load_renderings(self, config):
        """Load images from disk."""
        render_only = config.render_path and self.split == utils.DataSplit.TEST

        basedir = os.path.join(self.data_dir, 'dense')
        sizes = [f for f in sorted(utils.listdir(basedir)) if f.startswith('ibr3d')]
        sizes = sizes[::-1]

        if config.factor >= len(sizes):
            raise ValueError(f'Factor {config.factor} larger than {len(sizes)}')

        basedir = os.path.join(basedir, sizes[config.factor])
        open_fn = lambda f: utils.open_file(os.path.join(basedir, f), 'rb')

        files = [f for f in sorted(utils.listdir(basedir)) if f.startswith('im_')]
        if render_only:
            files = files[:1]
        images = np.array([np.array(Image.open(open_fn(f))) for f in files]) / 255.

        names = ['Ks', 'Rs', 'ts']
        intrinsics, rot, trans = (np.load(open_fn(f'{n}.npy')) for n in names)

        # Convert poses from colmap world-to-cam into our cam-to-world.
        w2c = np.concatenate([rot, trans[..., None]], axis=-1)
        c2w_colmap = np.linalg.inv(camera_utils.pad_poses(w2c))[:, :3, :4]
        c2w = c2w_colmap @ np.diag(np.array([1, -1, -1, 1]))

        # Reorient poses so z-axis is up
        poses, _, _ = camera_utils.transform_poses_pca(c2w)
        self.poses = poses

        self.images = images
        self.height, self.width = self.images.shape[1:3]
        self.camtoworlds = poses
        # For now, ignore all but the first focal length in intrinsics
        self.focal = intrinsics[0, 0, 0]
        self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                   self.height)

        if render_only:
            render_path = camera_utils.generate_ellipse_path(
                poses,
                config.render_path_frames,
                z_variation=config.z_variation,
                z_phase=config.z_phase)
            self.images = None
            self.camtoworlds = render_path
            self.render_poses = render_path
        else:
            # Select the split.
            all_indices = np.arange(images.shape[0])
            indices = {
                utils.DataSplit.TEST:
                    all_indices[all_indices % config.llffhold == 0],
                utils.DataSplit.TRAIN:
                    all_indices[all_indices % config.llffhold != 0],
            }[self.split]

            self.images = self.images[indices]
            self.camtoworlds = self.camtoworlds[indices]


class DTU(Dataset):
    """DTU Dataset."""

    def _load_renderings(self, config):
        """Load images from disk."""
        if config.render_path:
            raise ValueError('render_path cannot be used for the DTU dataset.')

        images = []
        pixtocams = []
        camtoworlds = []

        # Find out whether the particular scan has 49 or 65 images.
        n_images = len(utils.listdir(self.data_dir)) // 8

        # Loop over all images.
        for i in range(1, n_images + 1):
            # Set light condition string accordingly.
            if config.dtu_light_cond < 7:
                light_str = f'{config.dtu_light_cond}_r' + ('5000'
                                                            if i < 50 else '7000')
            else:
                light_str = 'max'

            # Load image.
            fname = os.path.join(self.data_dir, f'rect_{i:03d}_{light_str}.png')
            image = utils.load_img(fname) / 255.
            if config.factor > 1:
                image = lib_image.downsample(image, config.factor)
            images.append(image)

            # Load projection matrix from file.
            fname = os.path.join(self.data_dir, f'../../cal18/pos_{i:03d}.txt')
            with utils.open_file(fname, 'rb') as f:
                projection = np.loadtxt(f, dtype=np.float32)

            # Decompose projection matrix into pose and camera matrix.
            camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
            camera_mat = camera_mat / camera_mat[2, 2]
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot_mat.transpose()
            pose[:3, 3] = (t[:3] / t[3])[:, 0]
            pose = pose[:3]
            camtoworlds.append(pose)

            if config.factor > 0:
                # Scale camera matrix according to downsampling factor.
                camera_mat = np.diag([1. / config.factor, 1. / config.factor, 1.
                                      ]).astype(np.float32) @ camera_mat
            pixtocams.append(np.linalg.inv(camera_mat))

        pixtocams = np.stack(pixtocams)
        camtoworlds = np.stack(camtoworlds)
        images = np.stack(images)

        def rescale_poses(poses):
            """Rescales camera poses according to maximum x/y/z value."""
            s = np.max(np.abs(poses[:, :3, -1]))
            out = np.copy(poses)
            out[:, :3, -1] /= s
            return out

        # Center and scale poses.
        camtoworlds, _ = camera_utils.recenter_poses(camtoworlds)
        camtoworlds = rescale_poses(camtoworlds)
        # Flip y and z axes to get poses in OpenGL coordinate system.
        camtoworlds = camtoworlds @ np.diag([1., -1., -1., 1.]).astype(np.float32)

        all_indices = np.arange(images.shape[0])
        split_indices = {
            utils.DataSplit.TEST: all_indices[all_indices % config.dtuhold == 0],
            utils.DataSplit.TRAIN: all_indices[all_indices % config.dtuhold != 0],
        }
        indices = split_indices[self.split]

        self.images = images[indices]
        self.height, self.width = images.shape[1:3]
        self.camtoworlds = camtoworlds[indices]
        self.pixtocams = pixtocams[indices]


if __name__ == '__main__':
    from internal import configs
    import accelerate
    config = configs.Config()
    accelerator = accelerate.Accelerator()
    config.world_size = accelerator.num_processes
    config.local_rank = accelerator.local_process_index
    config.factor = 8
    dataset = WAYMO('train', 'zipnerf/data/waymo/0032150', config)
    print(len(dataset))
    for _ in tqdm(dataset):
        pass
    print('done')
    # print(accelerator.local_process_index)