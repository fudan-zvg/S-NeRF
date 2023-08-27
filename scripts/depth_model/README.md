# Depth Completion Pipeline

## Prepare for the environment

- Install basic packages
```shell
    pip install matplotlib tensorboard scipy einops opencv-python pypng 
    pip install nuscenes-devkit scikit-image pillow
```

- Install mseg
```shell
    cd ./external
    pip install -e ./mseg-semantic/mseg-api
    pip install -e ./mseg-semantic
```

- Build flow model
```shell
    cd ./SeparableFlow-main/libs/GANet
    TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
    python setup.py build
    cp -r build/lib* build/lib
```

## Download Pretrained models
- Put the pretrained models in the following path:
    1. [mseg-semantic/mseg_semantic/checkpoints/mseg-3m.pth](https://drive.google.com/file/d/1yoWAmmjJvDqTyGgUJQ-19CVnkH1mUZ7A/view?usp=sharing)
    2. [SeparableFlow-main/models/sepflow_universal.pth](https://drive.google.com/file/d/1x-Q0VvfBfdabFW-1KwRtTE8UiZeX4yZ7/view?usp=sharing)
    3. [Sparse-Depth-Completion/Saved/model_best_epoch.pth.tar](https://drive.google.com/file/d/181zn-qcuJp2z4k0SIK-qQV55TgrUisLS/view?usp=sharing)

## Run the pipeline
    1. Run `python scripts/run.py`
    2. The output will be saved in data/scenes/depths




