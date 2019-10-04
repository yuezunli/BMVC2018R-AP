DenseFoolbox
---
DenseFoolbox is a python repository for white-box attacking object detectors, instance segmentation. 
This repository is a simple implementation of our paper 
[Robust Adversarial Perturbation on Deep Proposal-based Models, BMVC2018.](https://www.albany.edu/~yl149995/papers/bmvc2018.pdf)  


### Content
1. [Overview](#Overview)
2. [Requirements](#Requirements)
3. [Demo](#Usage)

### Overview
![overview](https://www.albany.edu/~yl149995/imgs/bmvc2018/overview.png "overview")

We target Region Proposal Network (RPN) as the bottleneck of Deep-proposal based networks.
The detections can be disrupted by breaking object proposal generation. To do so, we disturb the 
predicted class score as well as offset regression of object proposals.


### Requirements
- Pytorch 0.4.0
- Ubuntu 16.04
- CUDA 8.0
- Python 2.7
- opencv3


### Demo 
#### Attacking Faster-RCNN
1. We use Faster-RCNN detector based on pytorch framework [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn).
We make modifications to this repository which can be downloaded [here](https://drive.google.com/open?id=1h6pJnb5DX54rTIorfCDvf3FMWyCTGF_p).
2. Unzip the repository to `object_detectors`.
3. Look into `attack_wrapper/object_detectors_v2` and run `run.py`.   
    ```commandline
    python run.py \
    --net=faster-rcnn \  # faster-rcnn or ssd (update later)
    --base=vgg16 \
    --data_dir=demo/ \
    --res_dir=res/
    ```
    
#### Attacking Mask-RCNN
Update later

### Citation
If you find this implementation helpful, please cite:

    @inproceedings{li2018rap,
	author={Li, Yuezun and Tian, Daniel and Chang, Mingching and Bian, Xiao and Lyu, Siwei},
	title={Robust Adversarial Perturbation on Deep Proposal-based Models},
	booktitle={BMVC},
	year={2018}}    
