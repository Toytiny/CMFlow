<div align="center">   

# Hidden Gems: 4D Radar Scene Flow Learning Using Cross-Modal Supervision
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2203.01137-b31b1b.svg)](https://arxiv.org/abs/2203.01137)
[![ ](https://img.shields.io/youtube/views/PjKgznDizhI?label=YouTube&style=flat)](https://www.youtube.com/watch?v=PjKgznDizhI&feature=youtu.be) 
![code visitors](https://visitor-badge.glitch.me/badge?page_id=Toytiny/CMFlow)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Toytiny/CMFlow/blob/main/LICENSE)

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
[[arXiv]](https://arxiv.org/pdf/2203.01137.pdf) [[demo]](https://youtu.be/PjKgznDizhI) [[page]](https://github.com/Toytiny/CMFlow)

<p align="left">
<img src='./src/openfig.png' width="500">
</p>

## Citation
If you find our work useful in your research, please consider citing:


```shell
@inproceedings{ding2023hidden,
  title={Hidden Gems: 4D Radar Scene Flow Learning Using Cross-Modal Supervision},
  author={Ding, Fangqiang and Palffy, Andras and Gavrila, Dariu M. and Lu, Chris Xiaoxuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1-10},
  year={2023} 
}
```
## Abstract 
This work proposes a novel approach to 4D radar-based scene flow estimation via cross-modal learning. Our approach is motivated by the co-located sensing redundancy in modern autonomous vehicles. Such redundancy implicitly provides various forms of supervision cues to the radar scene flow estimation. Specifically, we introduce a multi-task model architecture for the identified cross-modal learning problem and propose loss functions to opportunistically engage scene flow estimation using multiple cross-modal constraints for effective model training. Extensive experiments show the state-of-the-art performance of our method and demonstrate the effectiveness of cross-modal supervised learning to infer more accurate 4D radar scene flow. We also show its usefulness to two subtasks - motion segmentation and ego-motion estimation. 

## Method
| ![pipeline.jpg](src/pipeline_cvpr.png) | 
|:--:| 
| ***Figure 1. Cross-modal supervised learning pipeline for 4D radar scene flow estimation**. Given two consecutive radar point clouds as the input, the model architecture, which is composed of two stages (blue/orange block colours for stage 1/2), outputs the final scene flow together with the motion segmentation and a rigid ego-motion transformation. Cross-modal supervision signals retrieved from co-located modalities are utilized to constrain outputs with various loss functions. This essentially leads to a multi-task learning problem.* |

## Qualitative results
Here some GIFs are provided to show our qualitative results on scene flow estimation and two subtasks, motion segmentation and ego-motion estimation.
For more qualitative results, please refer to our [demo video](#demo-video) or paper.

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
<img src='./src/ego-motion/gif-1.gif' width="700">
<img src='./src/ego-motion/gif-2.gif' width="700">
</p>


## Demo Video 
<p align="center"> <a href="https://youtu.be/PjKgznDizhI"><img src="./src/cover.png" width="80%"></a> </p>