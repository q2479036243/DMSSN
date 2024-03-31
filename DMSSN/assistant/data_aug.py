#数据增强

import random
import numpy as np
import cv2
import torchvision.transforms as tf
from skimage import transform


def flip(image,mask): #水平翻转和垂直翻转
    if random.random()>0.5:
        image = np.flip(image, 1)
        mask = np.flip(mask, 1)
    if random.random()<0.5:
        image = np.flip(image, 2)
        mask = np.flip(mask, 2)
    return image, mask


def rotate_image(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]
    # if the center is None, initialize it as the center of
    if center is None:
        center = (w // 2, h // 2)
    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    # return the rotated image
    return rotated


def rotate(image,mask):
    angle = tf.RandomRotation.get_params([-180, 180]) # -180~180随机选一个角度旋转
    for i in range(image.shape[0]):
        image[i] = rotate_image(image[i], angle)
    for i in range(mask.shape[0]):
        mask[i] = rotate_image(mask[i], angle)
    return image, mask


def scale_up(image, mask):
    i_b, i_h, i_w = image.shape
    m_b, m_h, m_w = mask.shape
    sch = random.uniform(1.01,1.5)
    scw = random.uniform(1.01,1.5)
    ih = int(i_h * sch)
    iw = int(i_w * scw)
    image = transform.resize(image, (i_b, ih, iw))
    mask = transform.resize(mask, (m_b, ih, iw))
    rh = random.randrange(0,ih-i_h,1)
    rw = random.randrange(0,iw-i_w,1)
    image = image[:,rh:(rh+i_h),rw:(rw+i_w)]
    mask = mask[:,rh:(rh+i_h),rw:(rw+i_w)]
    return image, mask


def scale_down(image, mask):
    i_b, i_h, i_w = image.shape
    m_b, m_h, m_w = mask.shape
    sch = random.uniform(0.5,0.99)
    scw = random.uniform(0.5,0.99)
    ih = int(i_h * sch)
    iw = int(i_w * scw)
    image = transform.resize(image, (i_b, ih, iw))
    mask = transform.resize(mask, (m_b, ih, iw))
    rh = random.randrange(0,i_h-ih,1)
    rw = random.randrange(0,i_w-iw,1)
    image_ = np.zeros((i_b, i_h, i_w))
    mask_ = np.zeros((m_b, i_h, i_w))
    image_[:,rh:(rh+ih),rw:(rw+iw)] = image
    mask_[:,rh:(rh+ih),rw:(rw+iw)] = mask
    return image_, mask_


def scale(image, mask):
    if random.random()>0.5:
        image, mask = scale_down(image, mask)
    if random.random()<0.5:
        image, mask = scale_up(image, mask)
    return image, mask


def augmentation(img, label):
    img, label = scale(img, label)
    img, label = flip(img, label)
    img, label = rotate(img, label)
    
    return img, label