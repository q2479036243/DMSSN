# hlsod数据集的数据读取

from torch.utils.data import Dataset
from skimage import io, transform
import numpy as np
import scipy.io as scio
import sys
sys.path.append("..")
from data_prepare.sc_norm import *
from assistant.data_aug import *


def normalize_sc(data):
    for i in range(data.shape[0]):
        data[i,:,:] = (data[i,:,:] - mean_sc[i]) / std_sc[i]
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data


class HL_SC(Dataset):
    def __init__(self, img_list, img_path, lab_path, trans=True):
        self.image_list = img_list
        self.image_path = img_path
        self.label_path = lab_path
        self.trans = trans

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx].split("/")[-1]
        id = name.split(".")[-2]

        image_mat = scio.loadmat(self.image_path+name)
        image = image_mat['data']
        image = image.astype(np.float32)
        image_norm = normalize_sc(image)

        label = io.imread(self.label_path + id + '.jpg')
        label = transform.resize(label, (512, 512, 1))
        if np.max(label) > 1:
            label = (label - np.min(label)) / (np.max(label) - np.min(label))
        label = np.swapaxes(label, 2, 0)
        label = label.astype(np.float32)

        if self.trans:
            image_norm, label = augmentation(image_norm, label)
        
        img = image_norm.astype(np.float32)
        label = label.astype(np.float32)

        sample = {'image': img.copy(), 'label': label.copy(), 'id': id}

        return sample




class HL_RAW(Dataset):
    def __init__(self, img_list, img_path, lab_path, trans=True):
        self.image_list = img_list
        self.image_path = img_path
        self.label_path = lab_path
        self.trans = trans

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx].split("/")[-1]
        id = name.split(".")[-2]

        image_mat = scio.loadmat(self.image_path+name)
        image = image_mat['data']
        image = image.astype(np.float32)

        label = io.imread(self.label_path + id + '.jpg')
        label = transform.resize(label, (512, 512, 1))
        if np.max(label) > 1:
            label = (label - np.min(label)) / (np.max(label) - np.min(label))
        label = np.swapaxes(label, 2, 0)
        label = label.astype(np.float32)

        if self.trans:
            image, label = augmentation(image, label)
        
        img = image.astype(np.float32)
        label = label.astype(np.float32)

        sample = {'image': img.copy(), 'label': label.copy(), 'id': id}

        return sample
    


from sklearn.decomposition import PCA

class HL_SC_PCA(Dataset):
    def __init__(self, img_list, img_path, lab_path, trans=True):
        self.image_list = img_list
        self.image_path = img_path
        self.label_path = lab_path
        self.trans = trans

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx].split("/")[-1]
        id = name.split(".")[-2]

        image_mat = scio.loadmat(self.image_path+name)
        image = image_mat['data']
        image = image.astype(np.float32)
        image_norm = normalize_sc(image)

        pca = PCA(n_components=32)
        image_pca = np.swapaxes(image_norm, 2, 0)
        image_pca = image_pca.reshape(-1, 200)
        image_pca = pca.fit_transform(image_pca)
        image_pca = image_pca.reshape(512, 512, 32)
        image_pca = np.swapaxes(image_pca, 2, 0)
        image_norm = image_pca.astype(np.float32)

        label = io.imread(self.label_path + id + '.jpg')
        label = transform.resize(label, (512, 512, 1))
        if np.max(label) > 1:
            label = (label - np.min(label)) / (np.max(label) - np.min(label))
        label = np.swapaxes(label, 2, 0)
        label = label.astype(np.float32)

        if self.trans:
            image_norm, label = augmentation(image_norm, label)
        
        img = image_norm.astype(np.float32)
        label = label.astype(np.float32)

        sample = {'image': img.copy(), 'label': label.copy(), 'id': id}

        return sample
    


class HL_RAW_PCA(Dataset):
    def __init__(self, img_list, img_path, lab_path, trans=True):
        self.image_list = img_list
        self.image_path = img_path
        self.label_path = lab_path
        self.trans = trans

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx].split("/")[-1]
        id = name.split(".")[-2]

        image_mat = scio.loadmat(self.image_path+name)
        image = image_mat['data']
        image = image.astype(np.float32)

        pca = PCA(n_components=32)
        image_pca = np.swapaxes(image, 2, 0)
        image_pca = image_pca.reshape(-1, 200)
        image_pca = pca.fit_transform(image_pca)
        image_pca = image_pca.reshape(512, 512, 32)
        image_pca = np.swapaxes(image_pca, 2, 0)
        image = image_pca.astype(np.float32)

        label = io.imread(self.label_path + id + '.jpg')
        label = transform.resize(label, (512, 512, 1))
        if np.max(label) > 1:
            label = (label - np.min(label)) / (np.max(label) - np.min(label))
        label = np.swapaxes(label, 2, 0)
        label = label.astype(np.float32)

        if self.trans:
            image, label = augmentation(image, label)
        
        img = image.astype(np.float32)
        label = label.astype(np.float32)

        sample = {'image': img.copy(), 'label': label.copy(), 'id': id}

        return sample