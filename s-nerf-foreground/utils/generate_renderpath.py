import enum
import numpy as np
from pytorch3d.renderer.cameras import look_at_view_transform
from scipy.spatial.transform import Rotation as Rot

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


def generate_render_path(poses, focal, sc, N_views):
    dist = np.linalg.norm(poses[:,:3,3], 2, axis=1).mean()
    height = poses[:,1,3].mean()

    yaws = []
    for raw_pose in poses:
        x, y, z = raw_pose[:3,3]
        azi = np.math.atan2(-x,z)*180/np.pi
        yaws.append(azi)
    yaws = np.stack(yaws)
    min_yaw, max_yaw = yaws.min(), yaws.max()
    render_poses = []

    for theta in np.linspace(min_yaw, max_yaw, N_views):
        R, T = look_at_view_transform(dist, 0, theta, at=[(0,height,0)])
        r_pose = np.eye(4)
        r_pose[:3,:3] = R
        r_pose[:3,3] = T

        r_pose[0] = -r_pose[0]
        r_pose[2] = -r_pose[2]
        #import pdb; pdb.set_trace()
        r_pose = np.linalg.inv(r_pose)
        r_pose[:3,0] = -r_pose[:3,0]
        r_pose[:3,1] = -r_pose[:3,1]
        render_poses.append(r_pose)

    render_poses = np.stack(render_poses)
    
    return render_poses, focal, dist

def recenter_poses(poses):
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses, c2w