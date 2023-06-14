# S-NeRF: Neural Radiance Fields for Street Views
### [[Project]](https://ziyang-xie.github.io/s-nerf/)[ [Paper]](https://arxiv.org/abs/2303.00749) 

We introduce S-NeRF, a robust system to synthesizing large unbounded street views for autonomous driving using Neural Radiance Fields (NeRFs). This project aims to enhance the realism and accuracy of street view synthesis and improve the robustness of NeRFs for real-world applications. (e.g. autonomous driving simulation, robotics, and augmented reality)

## Key Features

- **Large-scale Street View Synthesis**: S-NeRF is able to synthesize large-scale street views with high fidelity and accuracy.

- **Improved Realism and Accuracy**: S-NeRF significantly improves the realism and accuracy of specular reflections and street view synthesis.

- **Robust Geometry and Reprojection**: By utilizing noisy and sparse LiDAR points, S-NeRF learns a robust geometry and reprojection based confidence to address the depth outliers.

- **Foreground Moving Vehicles**: S-NeRF extends its capabilities for reconstructing moving vehicles, a task that is impracticable for conventional NeRFs.

## Model Pipline Overview
![Model](./assets/model.png)

## Demo
[![Demo Video](https://img.youtube.com/vi/CY4NK-bvEus/0.jpg)](https://www.youtube.com/embed/CY4NK-bvEus)

## TODOs
- [x] Env Installation
- [ ] Pose Preparation Scripts
- [ ] Depth & Flow Preparation Scripts
- [ ] Code for training and testing
- [ ] Foreground Vehicle Reconstruction
- [ ] Pretrained models (Demo Scenes)

## Installation
Install the required packages:
```
pip install -r requiremnets.txt
```

### Data Preparation
1. prepare the poses, images and depth in S-NeRF format
- Waymo 
```
python waymo_preprocess.py
```

- nuScenes
```
python scripts/nuscenes_preprocess.py
```
generate the imgs and poses needed


### Training Snerf 
1. Dataset type: \
Waymo or nuScenes\
- Waymo:`5cams`, args.waymo = True\
- nuScenes:`6cams`, args.backcam = True

TODO


## Citation

If you find this work useful, please cite:
```
@inproceedings{ziyang2023snerf,
author = {Xie, Ziyang and Zhang, Junge and Li, Wenye and Zhang, Feihu and Zhang, Li},
title = {S-NeRF: Neural Radiance Fields for Street Views},
booktitle = {ICLR 2023},
year = {2023}
}
```



