<div align="center">   

# Getting Started
</div>

## 0. Data Download

First of all, please download the offical [View-of-Delft (VoD)](https://github.com/tudelft-iv/view-of-delft-dataset) dataset and keep the format of how the dataset is provided. Note that the VoD dataset is made freely available for non-commercial research purposes only. You need to request the access to the VoD dataset at first. 

The labels in the original release do not include track ids, please download the version with tracking ids and overwrite the original labels, following the official [instructions](https://github.com/tudelft-iv/view-of-delft-dataset/blob/main/docs/ANNOTATION.md#tracking-ids). In the end, the dataset should be oragnized like this:


```
View-of-Delft-Dataset (root)
    ├── radar (kitti dataset where velodyne contains the radar point clouds)
    │   │── ImageSets
    │   │── training
    │   │   ├──calib & velodyne & image_2 & label_2
    │   │── testing
    │       ├──calib & velodyne & image_2
    | 
    ├── lidar (kitti dataset where velodyne contains the LiDAR point clouds)
    ├── radar_3_scans (kitti dataset where velodyne contains the accumulated radar point clouds of 3 scans)
    ├── radar_5_scans (kitti dataset where velodyne contains the radar point clouds of 5 scans)
```

## 1. Installation

Before you run our code, please follow the steps below to build up your environment. 

a. Clone the repository to local
   
```
git clone https://github.com/Toytiny/CMFlow
```
b. Set up a new environment (Python 3.7)  with Anaconda 
   
```
conda create -n $ENV_NAME$ python=3.7
source activate $ENV_NAME$
```
c. Install common dependices 

```
pip install -r requirements.txt
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```
d. Install [PointNet++](https://github.com/sshaoshuai/Pointnet2.PyTorch) library for basic point cloud operation
```
cd lib
python setup.py install
cd ..
```

## 2. Data Preprocess

To run our radar scene flow experiments on the VoD dataset, you need to preprocess the original dataset into our scene flow format following all steps below.

a. Cope the official labels (with tracking ids) to this repository as `preprocess/label_track_gt`. 

Some other folders or files under `preprocess/` are:

**clips**: The official benchmark is for object detection and thus puts all frames together. Here, we provide the clip information (which frame belongs to which clip).

**scene_flow_clips_info.yaml**: As we split these clips ourselves into training, validation and testing sets, we organize this information as a yaml file.

**label_track_pre**: These are the prediction labels generated by running [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT) algorithms on the LiDAR point clouds of our training splits. These labels are used to provide cross-modal traininig supervision signals in our experiments.

b. Download the official [RAFT](https://github.com/princeton-vl/RAFT) pretrained model from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT). We use the raft-small model in our work to estimate optical flow from training images. This optical flow estimation can be used to provide cross-modal supervision singals in the image aspect. Please put the downloaded model as `preprocess/utils/RAFT/raft-small.pth`
    

c. Running the preprocessing code using:

```
python preprocess/preprocess_vod.py --root_dir $ROOT_DIR$ --save_dir $SAVE_DIR$
```

The final scene flow samples will be saved under the `$SAVE_DIR$/flow_smp/`. The preprocessing speed might be slow because we need to infer the optical flow results with the RAFT model for training samples. Each scene flow sample is a dictinary that includes:

```
#Key    Dimension      Description
----------------------------------------------------------------------------
   pc1    N×5        Source radar point clouds (x, y, z, RCS, doppler velocity).
   pc2    N×5    Target radar point clouds (x, y, z, RCS, doppler velocity).
   trans  4×4    The coordinate frame transformation between two frames
```

