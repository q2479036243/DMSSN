# raw数据保存，空间维度resize，光谱维度不变

import glob
from raw_norm import *
from skimage import transform
import os
import scipy.io as scio


data_dir = '/data3/QHL/DATA/SOD/HL-SOD/hyperspectral/'
dir_output = '/data3/QHL/HSOD/data_out/'
name_list = glob.glob(data_dir+'*.mat')
img_num = len(name_list)


def normalize_raw(data):
    for i in range(data.shape[0]):
        data[i,:,:] = (data[i,:,:] - mean_raw[i]) / std_raw[i]
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data


for i in range(img_num):
    name = name_list[i].split("/")[-1].split(".")[-2]
    image = h5py.File(name_list[i], "r")
    image = image["hypercube"].value.astype('float32')
    image_norm = normalize_raw(image)
    image_resize = transform.resize(image_norm, (200,512,512), preserve_range=True)
    path = os.path.join(dir_output, name + ".mat")
    scio.savemat(path, {'data':image_resize})
    print(i,name)
    