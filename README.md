# S-NeRF: Neural Radiance Fields for Street Views
### [[Project]](https://ziyang-xie.github.io/s-nerf/)[ [Paper]](https://arxiv.org/abs/2303.00749) 

## 👀 Foreground Vehicle Reconstruction Pipline

For reconstructing foreground elements, we provide access to both our training code and dataset. The dataset, which can be downloaded ![here](https://drive.google.com/drive/folders/1XgcjS7TmwKNY0FSELMajRr3q2MlHCk8U?usp=sharing), includes data on 16 moving vehicles and 64 static vehicles. Each vehicle is represented with 4 to 8 training views along with their corresponding depth information. The training code can be found in the s-nerf-foreground directory.

During the datacollection, the ego car (camera) is moving and the target car (object) is also moving. We propose a virtual camera transformation process that treats the target car (moving object) as static and then compute the relative camera poses for the ego car’s camera. These relative camera poses can be estimated through the 3D object
detectors. After the transformation, only the camera is moving which is favorable in training NeRFs.

![Model](./assets/model.png)

### 📂 Data Preparation
1. Download the data from this link [S-NeRF Data](https://drive.google.com/drive/folders/1XgcjS7TmwKNY0FSELMajRr3q2MlHCk8U?usp=sharing) and prepare the data in the following format

```
s-nerf-foreground/data/
├── nerf_data_moving/
└── nerf_data_static/
```

### 🚀 Train S-NeRF [Foreground]
```
cd s-nerf-foreground
python train.py --config [CONFIG FILE]
```


## 📝 Bibtex
If you find this work useful, please cite:
```bibtex
@inproceedings{ziyang2023snerf,
author = {Xie, Ziyang and Zhang, Junge and Li, Wenye and Zhang, Feihu and Zhang, Li},
title = {S-NeRF: Neural Radiance Fields for Street Views},
booktitle = {International Conference on Learning Representations (ICLR)},
year = {2023}
}
```



