# MFR-PGC-Net
This is the implementation of the paper "Utilizing Bounding Box Annotations for Weakly Supervised Building Extraction from Remote Sensing Images".

For more information, please checkout the [paper](https://ieeexplore.ieee.org/document/10113662).


## Requirements
* Python >= 3.6
* PyTorch >= 1.3.0
* yacs (https://github.com/rbgirshick/yacs)


## Getting started
The folder ```data``` should be like this
```
    datasets   
    └── WHU
        ├── train
        ├── BgMaskfromBoxes_train
        └── multi633_g3
            ├── Y_crf
            └── Y_ret
```


```bash
git https://gitee.com/labiao/mfr-pgc-net.git
cd MFR-PGC-Net
bash train_multi.sh # For training a classification network
# For transforming the weights of the Repvgg network to deploy.
python transform_to_deploy.py --NAME multi633_g3_deploy --config-file configs/grad_cam_repvgg.yml --WEIGHTS multi633_g3.pt 
bash generation_multi.sh # For generating pseudo labels
```


## Bibtex
```
@ARTICLE{10113662,
  author={Zheng, Daoyuan and Li, Shengwen and Fang, Fang and Zhang, Jiahui and Feng, Yuting and Wan, Bo and Liu, Yuanyuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Utilizing Bounding Box Annotations for Weakly Supervised Building Extraction From Remote-Sensing Images}, 
  year={2023},
  volume={61},
  number={},
  pages={1-17},
  doi={10.1109/TGRS.2023.3271986}}
```

## Acknowledgment

This code is heavily borrowed from [BANA](https://github.com/cvlab-yonsei/BANA), thanks!
