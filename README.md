# DMSSN

## Papaer

The paper "DMSSN: Distilled Mixed Spectral-Spatial Network for Hyperspectral Salient Object Detection" has been published and can be viewed at https://ieeexplore.ieee.org/abstract/document/10475351.

![overall](https://github.com/q2479036243/DMSSN/assets/54230421/aa5707d3-211c-4da7-92be-d34e21698bf8)


## Dataset

HSOD-BIT (V1), a large-scale dataset for hyperspectral saliency object detection, has been released. The improved version, HOSD-BIT (V2), which has more data and more comprehensive challenges, has been produced and will be released soon.

#### Description:

HSOD-BIT is the first large-scale, high-quality benchmark dataset for hyperspectral salient object detection, aimed at leveraging the advantages of spectral information to achieve higher precision in salient object detection tasks. Addressing the data requirements of contemporary deep learning models, this dataset provides pixel-level manual annotations for 319 hyperspectral data cubes and generates corresponding pseudo-color images. Each data cube contains 200 bands covering spectral information from visible light to near-infrared bands, with a spatial resolution of up to 1240×1680 pixels. In addition to conventional scenes, this dataset also specifically gathers challenging data to reflect the complexity of the real world, such as similar background interference, uneven lighting, overexposure, and other challenging scenarios. This further enhances the practicality and evaluation capabilities of the dataset.

![图片1](https://github.com/q2479036243/DMSSN/assets/54230421/5057ec2e-8110-4781-9382-d65cf8d08953)

#### Download:

Download link: https://pan.baidu.com/s/1AsdnO2-nadxTaq9_9Mo3Eg?pwd=tftf

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
