import sys
sys.path.append("..")
from assistant.evaluate_function import *
import glob
from skimage import io, transform
import numpy as np
from assistant.list import *

#data_dir = '/data3/QHL/HSOD/AT_out/'
# data_dir = '/data3/QHL/HSOD/basnet/output/'
# data_dir = '/data3/QHL/DMSSN/part/ICON/'
data_dir = '/data3/QHL/DMSSN/part/ICON-P/'
# gt_dir = '/data3/QHL/DATA/SOD/HL-SOD/ground_truth/'
gt_dir = '/data3/QHL/DMSSN/ground_truth/'

name_list = glob.glob(data_dir+'/'+'*.jpg')
# name_list = glob.glob(data_dir+'/'+'*.png')

# name_list = []
# test_list = test_lists
# for i in range(len(test_list)):
#     id = test_list[i].split("/")[-1].split(".")[-2]
#     name_list.append(data_dir + id + '.png')
#     # name_list.append(data_dir + id + '.jpg')

img_num = len(name_list)

mae, pre, rec, f_1, auc, cc, nss = 0, 0, 0, 0, 0, 0, 0
for i in range(img_num):
    name = name_list[i].split("/")[-1].split(".")[-2]
    image = io.imread(name_list[i])
    if len(image.shape) == 2:
        h,w = image.shape[0],image.shape[1]
        image = np.swapaxes(image, 1, 0)
        image = image.astype(np.float32)
    if len(image.shape) == 3:
        h,w,c = image.shape[0],image.shape[1],image.shape[2]
        image = np.swapaxes(image, 2, 0)
        image = image.astype(np.float32)
    label = io.imread(gt_dir + name + '.jpg')
    # print(image.shape,label.shape)
    # label = np.swapaxes(label, 1, 2)
    label = transform.resize(label, (h,w,3))
    label = np.swapaxes(label, 2, 0)
    label = label.astype(np.float32)
    mae_, pre_, rec_, f_1_, auc_, cc_, nss_ = evaluate(image, label)
    mae = mae + mae_
    pre = pre + pre_
    rec = rec + rec_
    f_1 = f_1 + f_1_
    auc = auc + auc_
    cc = cc + cc_
    nss = nss + nss_

len = img_num
mae = mae / len
pre = pre / len
rec = rec / len
f_1 = f_1 / len
auc = auc / len
cc = cc / len
nss = nss / len

print('pre, rec, f_1', pre, rec, f_1)
print('auc, cc, nss', auc, cc, nss)
print('mae',mae)