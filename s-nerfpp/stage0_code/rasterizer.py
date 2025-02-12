import os
import trimesh
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr


def projection(W, H, focal, near=0.1, far=1e5):
    return torch.tensor([[2 * focal / W, 0, 0, 0],
                            [0, 2 * focal / H, 0, 0],
                            [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
                            [0, 0, -1, 0]]).float()


def rasterize(mesh,
              resolution,
              intrinsic,
              v2c):
    
    W, H = resolution
    focal = (intrinsic[0, 0] + intrinsic[1, 1]) / 2
    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).to(torch.int32).cuda().contiguous()
    glctx = dr.RasterizeCudaContext()
    
    proj = projection(W, H, focal, near=0.1, far=1e5).cuda()
    v2c = torch.from_numpy(v2c[None]).float().cuda()
    vertices_homo = torch.cat([vertices, torch.ones(vertices.shape[0], 1).cuda()], dim=-1).float()
    vertices_camera = torch.matmul(vertices_homo, v2c.transpose(-2, -1))
    vertices_camera[..., 1] = -1 * vertices_camera[..., 1]
    vertices_camera[..., 0] = -1 * vertices_camera[..., 0]
    vertices_ndc = torch.matmul(vertices_camera, proj.T).contiguous()
    
    # ranges = torch.empty(size=(1, 2), dtype=torch.int32, device='cpu')
    rast, rast_db = dr.rasterize(glctx, vertices_ndc, faces, resolution=[H, W], ranges=None)
    
    mask = (rast != 0).any(dim=-1)
    uv_pts = dr.interpolate(vertices, rast, faces)[0]
    uv_pts_homo = torch.cat([uv_pts, torch.ones_like(uv_pts[..., :1]).cuda()], dim=-1).float()
    uv_pts_cam = torch.matmul(uv_pts_homo, v2c.transpose(-2, -1))
    depth = -uv_pts_cam[..., 2]    
    depth[~mask] = 0.
    
    mask = mask[0, ...].detach().cpu().numpy()
    mask = (mask.astype(np.uint8) * 255)[..., None].repeat(3, axis=-1)
    depth = depth[0, ...].detach().cpu().numpy()
    depth = np.clip(depth, 0, 256.)
    depth = (depth * 256.).astype(np.uint16)
    
    # import pdb; pdb.set_trace()
    
    return mask, depth