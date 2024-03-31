# 评价指标计算

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import Binarizer

def mask_normalize(mask):
    mask = (mask - np.min(mask))/(np.max(mask) - np.min(mask) + 1e-8)
    return mask

def compute_mae(mask,label):
    h, w = mask.shape[0], mask.shape[1]
    sumError = np.sum(np.absolute((mask.astype(float) - label.astype(float))))
    maeError = sumError/(float(h)*float(w)+1e-8)
    return maeError

def compute_p_r_f(mask,label):
    pre = precision_score(mask, label, average='binary')
    rec = recall_score(mask, label, average='binary')
    f_1 = f1_score(mask, label, average='binary')
    return pre, rec, f_1

def compute_auc(mask,label):
    auc = roc_auc_score(mask,label)
    return auc

def compute_cc(mask,label):
    mask = mask - np.mean(mask)
    label = label - np.mean(label)
    cov = np.sum(mask * label)
    d1 = np.sum(mask * mask)
    d2 = np.sum(label * label)
    cc = cov / (np.sqrt(d1) * np.sqrt(d2) + 1e-8)
    return cc

def compute_nss(mask,label):
    std=np.std(mask)
    u=np.mean(mask)
    mask=(mask-u)/std
    nss=mask*label
    nss=np.sum(nss)/np.sum(label)
    return nss

def evaluate(mask,label):
    if len(mask.shape)==3:
        mask = mask[0]
    if len(label.shape)==3:
        label = label[0]
    mask = mask_normalize(mask)
    label = mask_normalize(label)
    
    mae = compute_mae(mask,label)

    binarizer = Binarizer(threshold=0.5)
    mask_b = binarizer.transform(mask)
    label_b = binarizer.transform(label)
    mask_br = np.reshape(mask_b, newshape=(-1))
    label_br = np.reshape(label_b, newshape=(-1))

    pre, rec, f_1 = compute_p_r_f(mask_br,label_br)
    auc = compute_auc(mask_br,label_br)

    cc = compute_cc(mask,label)

    nss = compute_nss(mask,label_b)

    return mae, pre, rec, f_1, auc, cc, nss