# hlsod数据集，先进行光谱聚类，再resize并保存

from sklearn.mixture import GaussianMixture
import glob
from raw_norm import *
from skimage import transform
from skimage import measure
import numpy as np
import os
import scipy.io as scio


data_dir = '/data3/QHL/DATA/SOD/HL-SOD/hyperspectral/'
dir_output = '/data3/QHL/DMSSN/SC_out/'
name_list = glob.glob(data_dir+'*.mat')
img_num = len(name_list)


def normalize(data):
    if data.shape[0] == 200:
        for i in range(200):
            data[i,:,:] = (data[i,:,:] - mean_raw[i]) / std_raw[i]
    #data = data / np.max(data)
    return data


def gaussian_mixture_model(inp_image, n_clusters=100, iters=10):
    shape = inp_image.shape
    inp_image = inp_image.flatten().reshape(shape[0]*shape[1], shape[2])
    # Create Gaussian Mixture Model with Config Parameters
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type='full',
        max_iter=iters, random_state=21)
    # Fit on Input Image
    gmm.fit(X=inp_image)
    # Get Cluster Labels
    clust = gmm.predict(X=inp_image)

    return clust.reshape(shape[0], shape[1])


for i in range(img_num):
    name = name_list[i].split("/")[-1].split(".")[-2]
    image = h5py.File(name_list[i], "r")
    image = image["hypercube"].value.astype('float32')
    image_norm = normalize(image)
    image_resize = transform.resize(image_norm, (200,512,512), preserve_range=True)

    image_gmm = np.swapaxes(image_resize,0,2)
    gmm = gaussian_mixture_model(image_gmm,n_clusters=50,iters=5)
    label, sup = measure.label(gmm, background=0, return_num=True)

    for cla in range(sup):
        pos_list = np.where(label==sup)
        x_list = pos_list[0]
        y_list = pos_list[1]
        val = np.zeros([1,1,image_gmm.shape[2]])
        for num in range(len(x_list)):
            x = x_list[num]
            y = y_list[num]
            val = val + image_gmm[x:x+1,y:y+1,:]
        val = val / len(x_list)
        for num in range(len(x_list)):
            x = x_list[num]
            y = y_list[num]
            image_gmm[x:x+1,y:y+1,:] = val

    image_result = np.swapaxes(image_gmm,0,2)
    path = os.path.join(dir_output, name + ".mat")
    scio.savemat(path, {'data':image_result})
    print(i,name)