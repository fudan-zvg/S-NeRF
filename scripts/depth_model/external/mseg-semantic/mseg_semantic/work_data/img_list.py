import pickle
import os.path as osp
from PIL import Image

# f = open('../nuscenes/nuscenes_infos_train.pkl','rb')
f = open('./nuscenes_infos_train(2).pkl','rb')

data = pickle.load(f)

iters = len(data['infos'])

#* ins_imgpath_list = [[Img_Path * 4]*iters]
ins_imgpath_list = []
for i in range(iters):
    info = data['infos'][i]
    cams = info['cams']
    iter_imgpath_list = []
    for cam in cams:
        img_path = cam['data_path']
        iter_imgpath_list.append(img_path)
    ins_imgpath_list.append(iter_imgpath_list)

# import pdb; pdb.set_trace()

for iter_imgpath_list in ins_imgpath_list:
    for filename in iter_imgpath_list:
        im = Image.open(filename)
        filename = filename.split('/')[-1]
        im_path = osp.join('./img', filename)
        im.save(im_path)
    
