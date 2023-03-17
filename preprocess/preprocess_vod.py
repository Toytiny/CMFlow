import os
import cv2
import yaml
import torch
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.common import get_frame_list
from utils.vod.configuration import KittiLocations
from utils.get_flow_samples import get_radar_flow_samples


TASK = 'scene_flow'

def main(args):
    
    root_dir = args.root_dir
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # path for saving all scene flow samples
    smp_path = os.path.join(save_dir, 'flow_smp')
    # path for saving optical flow visulization (from RAFT)
    opt_path = os.path.join(save_dir, 'opt_vis')
    # pseudo_label_path are generated from AB3DMOT 
    # true_label_path are from the ground truth labels
    pseudo_label_path = 'preprocess/label_track_pre/'
    true_label_path = 'preprocess/label_track_gt/'
    data_loc = KittiLocations(root_dir=root_dir)

    clip_path = 'preprocess/clips/'

    with open('preprocess/' + TASK + '_split_info.yaml','r') as f:
        splits = yaml.safe_load(f.read())
    
    for split in splits:
        for clip in splits[split]:
            frames = get_frame_list(clip_path+'/'+clip+'.txt')

            # aggregate cross-modal info and consecutive radar pcs for training, validation and testing
            if split == 'train':
                get_radar_flow_samples(data_loc,frames,smp_path,opt_path, clip, split, pseudo_label_path, mode='train')
            if split == 'val':
                get_radar_flow_samples(data_loc,frames,smp_path,opt_path, clip, split, true_label_path, mode='val')
            if split == 'test':
                get_radar_flow_samples(data_loc,frames,smp_path,opt_path, clip, split, true_label_path, mode='test')
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--root_dir', type=str, default="/mnt/12T/fangqiang/view_of_delft/", help='Path for the origial dataset.')
    parser.add_argument('--save_dir', type=str, default='/mnt/12T/fangqiang/preprocess_res/', help='Path for saving preprocessing results.')
    args = parser.parse_args()
    main(args)


