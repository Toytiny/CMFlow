import os
import gc
from turtle import update
import cv2
import yaml
import torch
import argparse
import numpy as np
from utils.vod.frame import homogeneous_transformation, project_3d_to_2d
from utils.vod.visualization.settings import *
from utils.global_param import *
from utils.RAFT.core.raft import RAFT
from utils.RAFT.core.utils.flow_viz import flow_to_image


def estimate_optical_flow(img1,img2,model):

    resize_dim = (int(RESIZE_SCALE*img1.shape[1]),int(RESIZE_SCALE*img1.shape[0]))
    img1 = cv2.resize(img1,resize_dim)
    img2 = cv2.resize(img2,resize_dim)
    img1_torch = torch.from_numpy(img1).cuda().unsqueeze(0).transpose(1,3)
    img2_torch = torch.from_numpy(img2).cuda().unsqueeze(0).transpose(1,3)
    opt_flow = model(img1_torch, img2_torch, 12)
    np_flow = opt_flow.squeeze(0).permute(2,1,0).cpu().detach().numpy()
    resize_dim = (int(img1.shape[1]/RESIZE_SCALE),int(img1.shape[0]/RESIZE_SCALE))
    flow = cv2.resize(np_flow, resize_dim)

    return flow


def init_raft():

    # parser = argparse.ArgumentParser()
    
    # parser.add_argument('--model', default= "preprocess/utils/RAFT/raft-small.pth", help="restore checkpoint")
    # parser.add_argument('--path', help="dataset for evaluation")
    # parser.add_argument('--small', action='store_false', help='use small model')
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    # raft_args = parser.parse_args()
    raft_args = argparse.Namespace(model = "preprocess/utils/RAFT/raft-small.pth",\
                                    small = True,mixed_precision = False,alternate_corr = False)
    raft = RAFT(raft_args).cuda()
    raft = torch.nn.DataParallel(raft)
    raft.load_state_dict(torch.load(raft_args.model))

    return raft


def show_optical_flow(img1, img2, opt_flow, opt_path, frame1):

    flow_img = flow_to_image(opt_flow, convert_to_bgr=True)
    vis_img = np.concatenate((img1,img2,flow_img),axis=0)
    path = opt_path + '/' + frame1 + '.jpg'
    cv2.imwrite(path, vis_img)


def info_from_opt_flow(radar_data, transforms, opt_flow):

    radar_p = np.concatenate((radar_data[:,0:3],np.ones((radar_data.shape[0],1))),axis=1)
    radar_data_t = homogeneous_transformation(radar_p, transforms.t_camera_radar)
    uvs = project_3d_to_2d(radar_data_t,transforms.camera_projection_matrix)
    #filt_uv = np.logical_and(np.logical_and(uvs[:,0]>0, uvs[:,0]<opt_flow.shape[1]),\
    #     np.logical_and(uvs[:,1]>0, uvs[:,1]<opt_flow.shape[0]))

    radar_opt = opt_flow[uvs[:,1]-1,uvs[:,0]-1]

    opt_info = {"radar_u": uvs[:,0],
                "radar_v": uvs[:,1],
                "opt_flow": radar_opt,
                }

    return opt_info



def filt_points_in_fov(pc_data, transforms, sensor):

    pc_h = np.concatenate((pc_data[:,0:3],np.ones((pc_data.shape[0],1))),axis=1)
    if sensor == 'radar':
        pc_cam = homogeneous_transformation(pc_h, transforms.t_camera_radar)
    if sensor == 'lidar':
        pc_cam = homogeneous_transformation(pc_h, transforms.t_camera_lidar)
    uvs = project_3d_to_2d(pc_cam,transforms.camera_projection_matrix)
    filt_uv = np.logical_and(np.logical_and(uvs[:,0]>0, uvs[:,0]<=IMG_WIDTH),\
         np.logical_and(uvs[:,1]>0, uvs[:,1]<=IMG_HEIGHT))
    indices = np.argwhere(filt_uv).flatten()

    return indices
