#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import numpy as np
import ujson
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


class vodDataset(Dataset):

    def __init__(self, args, root='/mnt/12T/fangqiang/vod_unanno/flow_smp/', partition='train', textio=None):

        self.npoints = args.num_points
        self.textio = textio
        self.calib_path = 'dataset/vod_radar_calib.txt'
        self.res = {'r_res': 0.2, # m
                    'theta_res': 1.5 * np.pi/180, # radian
                    'phi_res': 1.5 *np.pi/180  # radian
                }
        self.read_calib_files()
        self.eval = args.eval
        self.partition = partition
        self.root = os.path.join(root, self.partition)
        self.interval = 0.10
        self.clips = sorted(os.listdir(self.root),key=lambda x:int(x.split("_")[1]))
        self.samples = []
        self.clips_info = []

        for clip in self.clips:
            clip_path = os.path.join(self.root, clip)
            samples = sorted(os.listdir(clip_path),key=lambda x:int(x.split("/")[-1].split("_")[0]))
            for idx in range(len(samples)):
                samples[idx] = os.path.join(clip_path, samples[idx])
            if self.eval:
                self.clips_info.append({'clip_name':clip, 
                                    'index': [len(self.samples),len(self.samples)+len(samples)]
                                })
            if clip[:5] == 'delft':
                self.samples.extend(samples)
        
        self.textio.cprint(self.partition + ' : ' +  str(len(self.samples)))
    

    def __getitem__(self, index):
        
        sample = self.samples[index]
        with open(sample, 'rb') as fp:
            data = ujson.load(fp)

        data_1 = np.array(data["pc1"]).astype('float32')
        data_2 = np.array(data["pc2"]).astype('float32')
        
        # read input data and features
        interval = self.interval
        pos_1 = data_1[:,0:3]
        pos_2 = data_2[:,0:3]
        feature_1 = data_1[:,[4,3,3]]
        feature_2 = data_2[:,[4,3,3]] 

        # GT labels and pseudo FG labels (from lidar)
        gt_labels = np.array(data["gt_labels"]).astype('float32')
        pse_labels = np.array(data["pse_labels"]).astype('float32')

        # GT mask or pseudo FG mask (from lidar)
        gt_mask = np.array(data["gt_mask"])
        pse_mask = np.array(data["pse_mask"])

        # use GT labels and motion seg. mask for evaluation on val and test set
        if self.partition in ['test','val', 'train_anno']:
            labels = gt_labels
            mask = gt_mask
            opt_flow =  np.zeros((pos_1.shape[0],2)).astype('float32')
            radar_u =  np.zeros(pos_1.shape[0]).astype('float32')
            radar_v =  np.zeros(pos_1.shape[0]).astype('float32')
        # use pseudo FG flow labels and FG mask as supervision signals for training 
        else:
            labels = pse_labels
            mask = pse_mask
            opt_info = data["opt_info"]
            opt_flow = np.array(opt_info["opt_flow"]).astype('float32')
            radar_u = np.array(opt_info["radar_u"]).astype('float32')
            radar_v = np.array(opt_info["radar_v"]).astype('float32')

        # static points transformation from frame 1 to frame 2  
        trans = np.linalg.inv(np.array(data["trans"])).astype('float32')

        ## downsample to npoints to enable fast batch processing (not in test)
        if not self.eval:
            
            npts_1 = pos_1.shape[0]
            npts_2 = pos_2.shape[0]

            ## if the number of points < npoints, fill empty space by duplicate sampling 
            ##  (filler points less than 25%)
            #if npts_1 < self.npoints * 0.75:
            #    raise('the number of points is lower than {}'.format(self.npoints * 0.75))
            if npts_1<self.npoints:
                sample_idx1 = np.arange(0,npts_1)
                sample_idx1 = np.append(sample_idx1, np.random.choice(npts_1,self.npoints-npts_1,replace=True))
            else:
                sample_idx1 = np.random.choice(npts_1, self.npoints, replace=False)
            if npts_2<self.npoints:
                sample_idx2 = np.arange(0,npts_2)
                sample_idx2 = np.append(sample_idx2, np.random.choice(npts_2,self.npoints-npts_2,replace=True))
            else:
                sample_idx2 = np.random.choice(npts_2, self.npoints, replace=False)
            
            pos_1 = pos_1[sample_idx1,:]
            pos_2 = pos_2[sample_idx2,:]
            feature_1 = feature_1[sample_idx1, :]
            feature_2 = feature_2[sample_idx2, :]
            radar_u = radar_u[sample_idx1]
            radar_v = radar_v[sample_idx1]
            opt_flow = opt_flow[sample_idx1,:]

            labels = labels[sample_idx1,:]
            mask = mask[sample_idx1]

        return pos_1, pos_2, feature_1, feature_2, trans, labels, mask, interval, radar_u, radar_v, opt_flow


    def read_calib_files(self):
        with open(self.calib_path, "r") as f:
            lines = f.readlines()
            intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Intrinsics
            extrinsic = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Extrinsic
            extrinsic = np.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)
        self.camera_projection_matrix = intrinsic
        self.t_camera_radar = extrinsic

    def __len__(self):
        return len(self.samples)
