import torch
import numpy as np
import cv2
from PIL import Image
def namedtuple_map(fn, tup):
    return type(tup)(*map(fn, tup))

# def shard(xs):
#   """Split data into shards for multiple devices along the first dimension."""
#   return jax.tree_map(
#       lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)
def shard(xs):
  """Split data into shards for multiple devices along the first dimension."""
  return xs.reshape((0, -1) + xs.shape[1:])

def unshard(x, padding=0):
  """Collect the sharded tensor to the shape before sharding."""
  y = x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))
  if padding > 0:
    y = y[:-padding]
  return y

def save_img_uint8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')

def change_brightness(img, value=30):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)
  v = cv2.add(v,value)
  v[v > 255] = 255
  v[v < 0] = 0
  final_hsv = cv2.merge((h, s, v))
  img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
  return img



def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    
    return c2w

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def generate_renderpath(poses, focal, N_views = 120, N_rots = 2, zrate=.5, sc=1.):
    '''
    poses: N x 3 x 4
    '''
    c2w = poses_avg(poses)

    up = normalize(poses[:, :3, 1].sum(0))

    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0) * sc

    render_poses = []
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * N_rots, N_views+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        pose = torch.tensor(viewmatrix(z, up, c))
        pose = torch.cat([pose, torch.tensor([[0,0,0,1]])], dim=0)
        render_poses.append(pose.float())
    return render_poses