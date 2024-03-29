<div align="center">   

# Hidden Gems: 4D Radar Scene Flow Learning Using Cross-Modal Supervision
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2303.00462-b31b1b.svg)](https://arxiv.org/abs/2303.00462)
[![ ](https://img.shields.io/youtube/views/PjKgznDizhI?label=Demo&style=flat)](https://www.youtube.com/watch?v=PjKgznDizhI&feature=youtu.be) 
[![GitHub license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Toytiny/CMFlow/blob/master/LICENSE)  [![GitHub](https://img.shields.io/website?label=Project%20Page&up_message=link&url=https://toytiny.github.io/publication/22-raflow-ral/index.html)](https://toytiny.github.io/publication/23-cmflow-cvpr/index.html)

<p align="center">
<img src='./src/scene-flow/row-1/gif-1.gif' width="400">
<img src='./src/scene-flow/row-1/gif-2.gif' width="400">
</p>


This is the official repository of the **CMFlow**, a cross-modal supervised approach for estimating 4D radar scene flow. 
For technical details, please refer to our paper on CVPR 2023:

**Hidden Gems: 4D Radar Scene Flow Learning Using Cross-Modal Supervision**
<br/>
[Fangqiang Ding](https://toytiny.github.io/), [Andras Palffy](https://scholar.google.co.uk/citations?user=_IIml4sAAAAJ&hl=en), [Dariu M. Gavrila](http://www.gavrila.net/), [Chris Xiaoxuan Lu](https://christopherlu.github.io/)
<br/>
[[arXiv]](https://arxiv.org/pdf/2303.00462.pdf) [[demo]](https://youtu.be/PjKgznDizhI) [[page]](https://toytiny.github.io/publication/23-cmflow-cvpr/index.html) [[supp]](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Ding_Hidden_Gems_4D_CVPR_2023_supplemental.pdf)  [[video]](https://youtu.be/hHgzBhy6NVQ)

<p align="left">
<img src='./src/openfig.png' width="500">
</p>

## News
 - [2023-02-28] Our paper is accepted by [CVPR 2023](https://cvpr2023.thecvf.com/) 🎉.
 - [2023-03-03] Our paper can be seen here 👉 [arXiv](https://arxiv.org/pdf/2303.00462.pdf). Supplementary can be found [here](https://drive.google.com/file/d/1Iewcqnjzecge2ePBM8k2tg-85LX5xs3N/view?usp=sharing). Project page is built [here](https://toytiny.github.io/publication/23-cmflow-cvpr/index.html). 
 - [2023-03-15] Our code has been released. Please see 👉 [GETTING_STARTED](./src/GETTING_STARTED.md) for the guidelines.
 - [2023-03-21] Our paper is selected as a highlight 🎉 in [CVPR 2023](https://cvpr2023.thecvf.com/) (10% of the accepted papers).
 - [2023-05-08] Our poster to CVPR 2023 is uploaded. Please download it 👉 [here](./src/poster_cvpr23_hidden_gems.pdf)
 - [2023-05-25] Our presentation video to CVPR 2023 is uploaded. Please watch it 👉 [here](https://youtu.be/hHgzBhy6NVQ)
 - [2023-07-18] We release our model trained with an extra amount of unlabeled data provided by the [VoD](https://github.com/tudelft-iv/view-of-delft-dataset/tree/main) dataset. Please try to following 👉 [MODEL_EVALUATION](https://github.com/Toytiny/CMFlow/blob/master/src/GETTING_STARTED.md#4-model-evaluation).
## Citation
If you find our work useful in your research, please consider citing:


```shell
@InProceedings{Ding_2023_CVPR,
    author    = {Ding, Fangqiang and Palffy, Andras and Gavrila, Dariu M. and Lu, Chris Xiaoxuan},
    title     = {Hidden Gems: 4D Radar Scene Flow Learning Using Cross-Modal Supervision},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {9340-9349}
}
```

## Getting Started

To find out how to run our scene flow experiments, please see our intructions in [GETTING_STARTED](./src/GETTING_STARTED.md). If you meet some issues when runinng our code, please raise them under this repository.


## Abstract 
This work proposes a novel approach to 4D radar-based scene flow estimation via cross-modal learning. Our approach is motivated by the co-located sensing redundancy in modern autonomous vehicles. Such redundancy implicitly provides various forms of supervision cues to the radar scene flow estimation. Specifically, we introduce a multi-task model architecture for the identified cross-modal learning problem and propose loss functions to opportunistically engage scene flow estimation using multiple cross-modal constraints for effective model training. Extensive experiments show the state-of-the-art performance of our method and demonstrate the effectiveness of cross-modal supervised learning to infer more accurate 4D radar scene flow. We also show its usefulness to two subtasks - motion segmentation and ego-motion estimation. 

## Method
| ![pipeline.jpg](src/pipeline_cvpr.png) | 
|:--:| 
| ***Figure 1. Cross-modal supervised learning pipeline for 4D radar scene flow estimation**. Given two consecutive radar point clouds as the input, the model architecture, which is composed of two stages (blue/orange block colours for stage 1/2), outputs the final scene flow together with the motion segmentation and a rigid ego-motion transformation. Cross-modal supervision signals retrieved from co-located modalities are utilized to constrain outputs with various loss functions. This essentially leads to a multi-task learning problem.* |

## Qualitative results
Here are some GIFs to show our qualitative results on scene flow estimation and two subtasks, motion segmentation and ego-motion estimation. For more qualitative results, please refer to our [demo video](#demo-video) or [supplementary](https://drive.google.com/file/d/1Iewcqnjzecge2ePBM8k2tg-85LX5xs3N/view?usp=sharing).

#### Scene flow

<p align="center">
<img src='./src/scene-flow/gif-1.gif' width="840">
<img src='./src/scene-flow/gif-2.gif' width="840">
</p>


#### Subtask - Motion Segmentation
<p align="center">
<img src='./src/motion-seg/gif-1.gif' width="840">
<img src='./src/motion-seg/gif-2.gif' width="840">
</p>

#### Subtask - Ego-motion Estimation
<p align="center">
<img src='./src/ego-motion/gif-1.gif' width="600">
<img src='./src/ego-motion/gif-2.gif' width="600">
</p>


## Demo Video 
<p align="center"> <a href="https://youtu.be/PjKgznDizhI"><img src="./src/cover.png" width="80%"></a> </p>

