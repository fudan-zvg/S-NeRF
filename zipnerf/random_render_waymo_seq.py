import glob
import os
import time

from absl import app
import gin
from internal import configs
from internal import datasets
from internal import models
from internal import train_utils
from internal import checkpoints
from internal import utils
from matplotlib import cm
import mediapy as media
import torch
import numpy as np
import accelerate
import imageio
from torch.utils._pytree import tree_map

configs.define_common_flags()



num_class = 19
def def_color_map():
    s = 256**3//num_class
    colormap = [[(i*s)//(256**2),((i*s)//(256)%256),(i*s)%(256) ] for i in range(num_class)]
    return colormap

color_map = np.array(def_color_map())


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        # acc = (TP + TN) / (TP + TN + FP + TN)
        Acc = np.diag(self.confusion_matrix).sum() / \
            self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        # acc = (TP) / TP + FP
        Acc = np.diag(self.confusion_matrix) / \
            self.confusion_matrix.sum(axis=1)
        Acc_class = np.nanmean(Acc)
        return Acc_class

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def measure_pa_miou(num_class, gt_image, pre_image):
    metric = Evaluator(num_class)
    metric.add_batch(gt_image, pre_image)
    acc = metric.Pixel_Accuracy()
    mIoU = metric.Mean_Intersection_over_Union()
    print("像素准确度PA:", acc, "平均交互度mIOU:", mIoU)
    return acc, mIoU



def visualize_depth(depth, near=0.2, far=13):
    colormap = cm.get_cmap('turbo')
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]
    # os.makedirs('./test_depths/'+pathname,exist_ok=True)
    out_depth = np.clip(np.nan_to_num(vis), 0., 1.) * 255
    return out_depth



def main(unused_argv):
    config = configs.load_config()
    config.exp_path = os.path.join(config.exp_name)
    config.render_dir = os.path.join(config.exp_path, 'render')

    accelerator = accelerate.Accelerator()
    config.world_size = accelerator.num_processes
    config.local_rank = accelerator.local_process_index

    ### set param for simnerf render
    config.dataset_loader = 'waymo_render'
    if config.demo:
        config.dataset_loader = 'waymo_demo'
    # config.RENDER_N =
    # config.only_side_cam = arg.only_side_cam
    # config.only_front_cam = arg.only_front_cam
    config.use_semantic = True


    dataset = datasets.load_dataset('test', config.data_dir, config)
    # if dataset.semantics is not None:
    #     config.use_semantic = True
    # else:
    #     config.use_semantic = False
    scale_factor = dataset.scale_factor

    utils.seed_everything(config.seed + accelerator.local_process_index)
    model = models.Model(config=config)

    step = checkpoints.restore_checkpoint(config.exp_path, model)
    accelerator.print(f'Rendering checkpoint at step {step}.')
    model.to(accelerator.device)


    dataloader = torch.utils.data.DataLoader(np.arange(len(dataset)),
                                             num_workers=0,
                                             shuffle=False,
                                             batch_size=1,
                                             collate_fn=dataset.collate_fn,
                                             )
    dataiter = iter(dataloader)
    if config.rawnerf_mode:
        postprocess_fn = dataset.metadata['postprocess_fn']
    else:
        postprocess_fn = lambda z: z

    out_name = 'path_renders' if config.render_path else 'test_preds'
    out_name = f'{out_name}_step_{step}'
    # out_dir = os.path.join(config.render_dir, out_name)
    out_dir = os.path.join(config.root_dir, config.wkdir, 'raw_data', 'background', config.result_name)

    if not utils.isdir(out_dir):
        utils.makedirs(out_dir)
    utils.makedirs(os.path.join(out_dir, 'rgb'))
    utils.makedirs(os.path.join(out_dir, 'depth'))
    utils.makedirs(os.path.join(out_dir, 'semantic'))
    utils.makedirs(os.path.join(out_dir, 'paint'))




    # args = config

    path_fn = lambda x: os.path.join(out_dir, x)

    np.save(path_fn('raw_target_poses.npy'), dataset.raw_poses)
    np.save(path_fn('intrinsic.npy'), dataset.intrinsic_fwd)
    np.save(path_fn('render_poses.npy'), dataset.render_poses_sd)

    # Ensure sufficient zero-padding of image indices in output filenames.
    zpad = max(5, len(str(dataset.size - 1)))
    idx_to_str = lambda idx: str(idx).zfill(zpad)
    seg_acc, seg_miou = [], []
    for idx in range(dataset.size):
        # If current image and next image both already exist, skip ahead.
        idx_str = idx_to_str(idx)
        # curr_file = path_fn(f'rgb/{idx_str}.png')
        # if utils.file_exists(curr_file):
        #     accelerator.print(f'Image {idx + 1}/{dataset.size} already exists, skipping')
        #     continue
        batch = next(dataiter)
        batch['semantic'] = batch['origins'].clone()
        batch = tree_map(lambda x: x.to(accelerator.device) if x is not None else None, batch)
        accelerator.print(f'Evaluating image {idx + 1}/{dataset.size}')
        eval_start_time = time.time()
        rendering = models.render_image(
            lambda rand, x: model(rand,
                                  x,
                                  train_frac=1.,
                                  compute_extras=True,
                                  sample_n=config.sample_n_test,
                                  sample_m=config.sample_m_test,
                                  ),
            accelerator,
            batch, False, config)

        accelerator.print(f'Rendered in {(time.time() - eval_start_time):0.3f}s')

        if accelerator.is_local_main_process:  # Only record via host 0.
            rendering['rgb'] = postprocess_fn(rendering['rgb'])
            rendering = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, rendering)
            utils.save_img_u8(rendering['rgb'], path_fn(f'rgb/{idx_str}.png'))
            if 'normals' in rendering:
                utils.save_img_u8(rendering['normals'] / 2. + 0.5,
                                  path_fn(f'normals_{idx_str}.png'))
            dep = rendering['depth']
            dep = (dep * 256/scale_factor).astype(np.uint16)
            from PIL import Image
            Image.fromarray(dep).save(path_fn(f'depth/{idx_str}.png'))
            logits_2_label = lambda x: np.argmax(x, axis=-1)
            labels = logits_2_label(rendering['semantic'])


            labels_color = color_map[labels]
            Image.fromarray(labels.astype(np.uint8)).save(path_fn(f'semantic/{idx_str}.png'))
            Image.fromarray(labels_color.astype(np.uint8)).save(path_fn(f'paint/{idx_str}.png'))
    accelerator.wait_for_everyone()


if __name__ == '__main__':
    with gin.config_scope('eval'):  # Use the same scope as eval.py
        app.run(main)