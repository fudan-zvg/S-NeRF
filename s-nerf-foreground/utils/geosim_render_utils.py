from math import degrees
from tkinter import Canvas
import torch
import torch.nn as nn
import copy
import numpy as np
from pytorch3d.structures import Meshes, join_meshes_as_batch
import os
import imageio
from pytorch3d.io import load_obj, save_obj
from mmdet3d.core.visualizer import show_result
from mmdet3d.core.bbox import Box3DMode
from pytorch3d.renderer import (
    PerspectiveCameras,
    OrthographicCameras,
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.renderer import (
    PointLights, 
    RasterizationSettings, 
)
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F


raster_settings_soft = RasterizationSettings(
    image_size=(1200,1600), 
    blur_radius=0,
    faces_per_pixel=1, 
)

class MeshDepthRenderer(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer and shader class which each have a forward
    function.
    """

    def __init__(self, rasterizer, shader) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)
        return self

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.

        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        For this set rasterizer.raster_settings.clip_barycentric_coords=True
        """
        fragments = self.rasterizer(meshes_world, **kwargs)
        #images = self.shader(fragments, meshes_world, **kwargs)

        return fragments.zbuf


def load_mesh(target_mesh):
    verts, faces, _ = load_obj(target_mesh)
    mesh = Meshes(verts=[verts], faces=[faces.verts_idx]).cuda()
    return mesh

def render_mesh_depth(mesh, cam_intr, cam_tran, cam_rot):
    focal_length = [(cam_intr[0][0], cam_intr[1][1])]
    principle_point = [(cam_intr[0][2], cam_intr[1][2])]

    camera = PerspectiveCameras(
        device='cuda:0', 
        focal_length=focal_length,
        principal_point=principle_point,
        R=cam_rot.unsqueeze(0), 
        T=cam_tran.unsqueeze(0),
        in_ndc=False,
        image_size=[(1200,1600)])

    renderer_depth = MeshDepthRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings_soft
        ),
        shader=SoftSilhouetteShader()
    )
    
    depth_img = renderer_depth(mesh, cameras=camera)

    depth_img = depth_img[0]
    depth_img = depth_img[0:900, :, 0]
    return depth_img
