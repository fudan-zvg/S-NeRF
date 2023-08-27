import os
import json

SENSORS = [
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT'
]
        
data_path = '../../data/scene_dict.json'
with open(data_path, 'r') as f:
    scenes = json.load(f)
os.makedirs('output', exist_ok=True)

for scene_name, scene_token in scenes.items():
    cmd = f"python -u ./scripts/run_pipeline.py --scene_name {scene_name}"
    flag = os.system(cmd)
    if flag != 0:
        # Sth Wrong
        import pdb; pdb.set_trace()

    IDX = 0
    depth_path = f'../../data/scenes/{scene_name}/depths'
    os.makedirs(depth_path, exist_ok=True)
    for sensor in SENSORS:
        path = f'./output/6cam_depth_data/{sensor}/'
        for file_name in sorted(os.listdir(path)):
            file_path = os.path.join(path, file_name)
            cmd = f'cp {file_path} {depth_path}/{IDX:04d}.png'
            os.system(cmd)
            IDX += 1
    