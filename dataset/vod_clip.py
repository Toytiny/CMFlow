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


class vodClipDataset(Dataset):

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
        self.mini_clip_len = args.mini_clip_len
        self.update_len = args.update_len
        self.clips = sorted(os.listdir(self.root),key=lambda x:int(x.split("_")[1]))
        self.mini_samples = []
        self.samples = []
        self.clips_info = []
        self.mini_clips_info = []
        
        for clip in self.clips:
            clip_path = os.path.join(self.root, clip)
            samples = sorted(os.listdir(clip_path),key=lambda x:int(x.split("/")[-1].split("_")[0]))

            if self.eval:
                self.clips_info.append({'clip_name':clip, 
                                    'index': [len(self.samples),len(self.samples)+len(samples)]
                                })
            
                for j in range(len(samples)):
                     self.samples.append(os.path.join(clip_path, samples[j]))

            if not self.eval: 
                clip_num = int(np.floor(len(samples)/self.mini_clip_len))
                ## take mini_clip as a sample
                for i in range(clip_num):
                    st_idx = i * self.mini_clip_len
                    mini_sample = []
                    for j in range(self.mini_clip_len):
                        mini_sample.append(os.path.join(clip_path, samples[st_idx+j]))
                        self.samples.append(os.path.join(clip_path, samples[st_idx+j]))
                    self.mini_samples.append(mini_sample)

        if not self.eval:
            self.textio.cprint(self.partition + ' : ' +  str(len(self.mini_samples)) + ' mini_clips')
        if self.eval:
            self.textio.cprint(self.partition + ' : ' +  str(len(self.samples)) + ' frames')

    def __getitem__(self, index):
        
        if not self.eval:
            return self.get_clip_item(index)
        if self.eval:
            with open(self.samples[index], 'rb') as fp:
                data = ujson.load(fp)
            return self.get_sample_item(data)


    def get_sample_item(self, data):

        data_1 = np.array(data["pc1"]).astype('float32')
        data_2 = np.array(data["pc2"]).astype('float32')
            
        # read input data and features
        interval = self.interval
        pos_1 = data_1[:,0:3]
        pos_2 = data_2[:,0:3]
        feature_1 = data_1[:,[4,3,3]]
        feature_2 = data_2[:,[4,3,3]] 

        # GT labels and pseudo labels (from lidar)
        gt_labels = np.array(data["gt_labels"]).astype('float32')
        pse_labels = np.array(data["pse_labels"]).astype('float32')

        # GT mask or pseudo mask (from lidar)
        gt_mask = np.array(data["gt_mask"])
        pse_mask = np.array(data["pse_mask"])

        # use GT labels and mask for evaluation on val and test set
        if self.partition in ['test','val', 'train_anno']:
            labels = gt_labels
            mask = gt_mask
            opt_flow =  np.zeros((pos_1.shape[0],2)).astype('float32')
            radar_u =  np.zeros(pos_1.shape[0]).astype('float32')
            radar_v =  np.zeros(pos_1.shape[0]).astype('float32')
        # use pseudo labels and mask as supervision signals for training 
        else:
            labels = pse_labels
            mask = pse_mask
            opt_info = data["opt_info"]
            opt_flow = np.array(opt_info["opt_flow"]).astype('float32')
            radar_u = np.array(opt_info["radar_u"]).astype('float32')
            radar_v = np.array(opt_info["radar_v"]).astype('float32')

        # static points transformation from frame 1 to frame 2  
        trans = np.linalg.inv(np.array(data["trans"])).astype('float32')
        
        ## keep npoints to enable fast batch processing (not in test)
        if not self.eval:

            sample_idx1, sample_idx2 = self.sample_points(pos_1.shape[0], pos_2.shape[0])
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


    def get_clip_item(self, index):

        mini_sample = self.mini_samples[index]
        mini_pos_1 = np.zeros((self.mini_clip_len, self.npoints,3)).astype('float32')
        mini_pos_2 = np.zeros((self.mini_clip_len, self.npoints,3)).astype('float32')
        mini_feat_1 = np.zeros((self.mini_clip_len, self.npoints,3)).astype('float32')
        mini_feat_2 = np.zeros((self.mini_clip_len, self.npoints,3)).astype('float32') 
        mini_trans = np.zeros((self.mini_clip_len, 4, 4)).astype('float32')
        mini_labels = np.zeros((self.mini_clip_len, self.npoints, 3)).astype('float32')
        mini_mask = np.zeros((self.mini_clip_len, self.npoints)).astype('float32')
        mini_interval = np.zeros((self.mini_clip_len)).astype('float32')
        mini_radar_u = np.zeros((self.mini_clip_len, self.npoints)).astype('float32')
        mini_radar_v = np.zeros((self.mini_clip_len, self.npoints)).astype('float32')
        mini_opt_flow = np.zeros((self.mini_clip_len, self.npoints, 2)).astype('float32')

        for i in range(0,len(mini_sample)):
            with open(mini_sample[i], 'rb') as fp:
                data = ujson.load(fp)

            pos_1, pos_2, feature_1, feature_2,trans, labels, mask,\
                 interval, radar_u, radar_v, opt_flow = self.get_sample_item(data) 

            # accumulate sample information
            mini_pos_1[i] = pos_1
            mini_pos_2[i] = pos_2
            mini_feat_1[i] = feature_1
            mini_feat_2[i] = feature_2
            mini_trans[i] = trans
            mini_labels[i] = labels
            mini_mask[i] = mask
            mini_interval[i] = interval
            mini_radar_u[i] = radar_u
            mini_radar_v[i] = radar_v
            mini_opt_flow[i] = opt_flow

        return mini_pos_1, mini_pos_2, mini_feat_1, mini_feat_2, mini_trans, mini_labels, mini_mask,\
            mini_interval, mini_radar_u, mini_radar_v, mini_opt_flow

    def read_calib_files(self):
        with open(self.calib_path, "r") as f:
            lines = f.readlines()
            intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Intrinsics
            extrinsic = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Extrinsic
            extrinsic = np.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)
        self.camera_projection_matrix = intrinsic
        self.t_camera_radar = extrinsic

    def sample_points(self, npts_1, npts_2,):

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
        return sample_idx1, sample_idx2

    def __len__(self):

        if not self.eval:
            return len(self.mini_samples)
        if self.eval:
            return len(self.samples)
