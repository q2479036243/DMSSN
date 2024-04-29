# DMSSN

## Papaer

The paper "DMSSN: Distilled Mixed Spectral-Spatial Network for Hyperspectral Salient Object Detection" has been published and can be viewed at https://ieeexplore.ieee.org/abstract/document/10475351.

![overall](https://github.com/q2479036243/DMSSN/assets/54230421/aa5707d3-211c-4da7-92be-d34e21698bf8)


## Dataset

HSOD-BIT is the first large-scale, high-quality benchmark dataset for hyperspectral salient object detection, please see https://github.com/anonymous0519/HSOD-BIT for more details.

## Code

1、Refer to requirements.txt to install dependent environments.

2、Run data_prepare/sc_demo.py and data_prepare/sc_norm.py for data preprocessing.

3、Train the teacher network.

nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=6666 knowledge/teacher_train.py > teacher_train.log 2>&1 &

4、Train the DMSSN.

nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=6666 tools/train.py > DMSSN.log 2>&1 &

## Important Update

For more efficient data storage, the hyperspectral image data format is changed from the original MAT to H5.

```python
import h5py
def dataload(path):
    data = h5py.File(save_name, "r")['dataset'][:]
    return data
```

## Citation

If you use this benchmark in your research, please cite this project.

```
@article{qin2024dmssn,
  title={DMSSN: Distilled Mixed Spectral-Spatial Network for Hyperspectral Salient Object Detection},
  author={Qin, Haolin and Xu, Tingfa and Liu, Peifu and Xu, Jingxuan and Li, Jianan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```
