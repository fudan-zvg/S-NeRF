import math
import nvdiffrast
import numpy as np
import torch
from torchvision.utils import save_image
import kaolin as kal
import os
import yaml
import argparse
from tqdm import tqdm
import trimesh


def center_mesh_bottom(mesh, scale=1, category='vehicle'):
    vertices = mesh.vertices.unsqueeze(0).cuda()
    vertices_min = vertices.min(dim=1, keepdims=True)[0]
    vertices_max = vertices.max(dim=1, keepdims=True)[0]
    vertices -= (vertices_max + vertices_min) / 2.
    l0 = (vertices_max - vertices_min).max()
    l = l0
    if category == 'vehicle':
        l = 4.5+(np.random.random()-0.5)*0.2
    if category == 'person':
        l = 1.7
    if category == 'bicycle':
        l = 1.6
    if category == 'motorcycle':
        l = 2.

    scale = l/l0
    bot = vertices[...,1].min()
    vertices[...,1] = vertices[...,1]-bot
    vertices = vertices*scale
    return vertices

def render(cam):
    transformed_vertices = cam.transform(vertices)
    # Create a fake W (See nvdiffrast documentation)
    pos = torch.nn.functional.pad(
        transformed_vertices, (0, 1), mode='constant', value=1.
    ).contiguous()
    rast = nvdiffrast.torch.rasterize(glctx, pos, faces.int(), (height, width), grad_db=False)
    hard_mask = rast[0][:, :, :, -1:] != 0
    def valid_mask(mask):
        l,r,t,b = mask[0,0].max(),mask[0,-1].max(),mask[0,:,0].max(),mask[0,:,-1].max()
        if l and r:
            return False
        if t and b:
            return False

        if mask.sum()/width/height > 0.5:
            return False
        return True
    if not valid_mask(hard_mask):
        hard_mask = torch.zeros_like(hard_mask, device=hard_mask.device) != 0


    if VERTEX_COLOR:
        out, _ = nvdiffrast.torch.interpolate(vertex_color[None], rast[0], faces.int())
        img = out
        return torch.flip(torch.clamp(img * hard_mask, 0., 1.)[0], dims=(0,)), \
               torch.flip(torch.clamp(torch.ones_like(img, device=img.device) * hard_mask, 0., 1.)[0], dims=(0,))


    face_idx = (rast[0][..., -1].long() - 1).contiguous()

    uv_map = nvdiffrast.torch.interpolate(uvs, rast[0], face_uvs_idx.int())[0]

    img = torch.zeros((1, height, width, 3), dtype=torch.float, device='cuda')

    # Obj meshes can be composed of multiple materials
    # so at rendering we need to interpolate from corresponding materials
    im_material_idx = face_material_idx[face_idx]
    im_material_idx[face_idx == -1] = -1

    for i, material in enumerate(materials):
        mask = im_material_idx == i
        mask_idx = torch.nonzero(mask, as_tuple=False)
        _texcoords = uv_map[mask] * 2. - 1.
        _texcoords[:, 1] = -_texcoords[:, 1]
        pixel_val = torch.nn.functional.grid_sample(
            materials[i], _texcoords.reshape(1, 1, -1, 2),
            mode='bilinear', align_corners=False,
            padding_mode='border')
        img[mask] = pixel_val[0, :, 0].permute(1, 0)

    return torch.flip(torch.clamp(img * hard_mask, 0., 1.)[0], dims=(0,)),\
           torch.flip(torch.clamp(torch.ones_like(img, device=img.device) * hard_mask, 0., 1.)[0], dims=(0,))

def get_theta_matrix(theta=0, shift=(0,0,-10)):
    cos = math.cos(theta)
    sin = math.sin(theta)
    return np.array([[cos,0,sin,shift[0]],
                  [0,1,0,shift[1]],
                  [-sin,0,cos,shift[2]],
                  [0,0,0,1]])


from collections import namedtuple
return_type = namedtuple('return_type',
                         ['vertices', 'faces', 'materials',
                          'vertex_color'])
def load_ply(path):
    mesh = trimesh.load_mesh(path)
    vert = mesh.vertices
    vert = torch.tensor(vert).float()
    vert = torch.stack([vert[:,0], vert[:,2], vert[:,1]], dim=1)
    face = mesh.faces
    v_attr = mesh.metadata['_ply_raw']['vertex']['data']
    color = [[v_attr[i][3]/255, v_attr[i][4]/255, v_attr[i][5]/255] for i in range(v_attr.shape[0])]
    color = torch.tensor(color)

    return return_type(vert, torch.tensor(face), [], color.float())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        default='experiments/version1/car/000002.obj_black/mesh/mesh.obj',)
    parser.add_argument('--camera_path', type=str,
                        default='wkdir_6/raw_data/background/dvgo_0570155/target_poses.npy')
    parser.add_argument('--intrinsic_path', type=str,
                        default='wkdir_6/raw_data/background/dvgo_0570155/intrinsic.npy')
    parser.add_argument('--meta_path', type=str,
                        default='wkdir_6/raw_data/foreground/vehicle_0/meta_data.yaml')
    parser.add_argument('--save_path', type=str,
                        default='test_api')
    parser.add_argument('--render_factor', type=int, default=4)
    parser.add_argument('--category', type=str, default='vehicle')
    parser.add_argument('--dataset', type=str, default='waymo')

    args = parser.parse_args()

    glctx = nvdiffrast.torch.RasterizeGLContext(False, device='cuda')

    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'mask'), exist_ok=True)

    OBJ_PATH = args.ckpt_path
    if OBJ_PATH[-3:] == 'obj':
        VERTEX_COLOR = False
        mesh = kal.io.obj.import_mesh(OBJ_PATH, with_materials=True)  # , triangulate=True)
    else:
        VERTEX_COLOR = True
        mesh = load_ply(OBJ_PATH)





    vertices = center_mesh_bottom(mesh, scale=4, category=args.category)
    faces = mesh.faces.cuda()

    if not VERTEX_COLOR:
        uvs = torch.nn.functional.pad(mesh.uvs.unsqueeze(0).cuda(), (0, 0, 0, 1)) % 1.
        face_uvs_idx = mesh.face_uvs_idx.cuda()
        face_material_idx = torch.zeros_like(face_uvs_idx)[:, 0].cuda()

        materials = [m['map_Kd'].permute(2, 0, 1).unsqueeze(0).cuda().float() / 255. if 'map_Kd' in m else
                     m['Kd'].reshape(1, 3, 1, 1).cuda()
                     for m in mesh.materials]
        nb_faces = faces.shape[0]

        mask = face_uvs_idx == -1
        face_uvs_idx[mask] = uvs.shape[1] - 1
    else:
        vertex_color = mesh.vertex_color.cuda()


    cameralist_path = args.camera_path
    intrinsic_path = args.intrinsic_path

    with open(args.meta_path, 'r', encoding='utf8') as file:
        default_yaml = yaml.safe_load(file)
    trace_path = np.array(default_yaml['world_coord_list'])
    angle_list = np.array(default_yaml['base_angle_list'])
    P_path = default_yaml['base_angle_list']
    cam_list = np.load(cameralist_path)
    intrinsic_list = np.load(intrinsic_path)
    if args.dataset == 'waymo':
        width, height = 1920, 1280
    else:
        width, height = 1600, 900

    def get_cam_idx(idx):
        if len(intrinsic_list.shape) == 2:
            focal_x = intrinsic_list[0, 0]
            x1, y1 = intrinsic_list[0, 2], intrinsic_list[1, 2]
        else:
            focal_x = intrinsic_list[idx, 0, 0]
            x1, y1 = intrinsic_list[idx, 0, 2], intrinsic_list[idx, 1, 2]

        # x0, y0 = -(x1 - width // 2), - (y1 - height // 2)
        x0, y0 = (x1 - width // 2), - (y1 - height // 2)

        view_matrix0 = cam_list[idx]
        view_matrix0[:3, 3] = view_matrix0[:3, 3] - trace_path[idx]
        left = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])

        left2 = np.array([[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])

        view_matrix = left2 @ np.linalg.inv(left @ view_matrix0)
        theta0 = math.atan2(view_matrix[0,2], view_matrix[0,0])
        matrix = get_theta_matrix((math.pi*(angle_list[idx]+180)/180-theta0), view_matrix[:3,3])
        matrix = torch.tensor(matrix)
        # import pdb; pdb.set_trace()


        cam = kal.render.camera.Camera.from_args(view_matrix=matrix,
                                                 focal_x=focal_x,
                                                 x0=x0, y0=y0,
                                                 width=width, height=height, device='cuda')
        return cam
    # y --> z, x --> -x, z --> y
    vertices_export = vertices[0]+0
    vertices_export = torch.cat([-vertices_export[:,0:1], vertices_export[:,2:], vertices_export[:,1:2]], -1)
    mesh = trimesh.Trimesh(vertices_export.cpu().numpy(), faces.cpu().numpy())
    mesh.export(os.path.join(save_dir, 'mesh.ply'))

    for i in tqdm(range(cam_list.shape[0])):
        cam = get_cam_idx(i)
        img, mask = render(cam)

        save_image(img.permute(2, 0, 1), '{}/{}/{:0>6d}.png'.format(save_dir, 'image', i))
        save_image(mask[...,0], '{}/{}/{:0>6d}.png'.format(save_dir, 'mask', i))


