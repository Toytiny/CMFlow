import os
import argparse
import sys
import torch
from tqdm import tqdm
import cv2
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.spatial.transform import Rotation as R
import matplotlib.ticker as ticker
from utils.vis_ops import flow_xy_to_colors

          
def visulize_result_2D_pre(pc1, pc2, pred_f, pc1_warp, gt, num_pcs, mask, args):

    pc_1=pc1[0].cpu().numpy()
    pc_2=pc2[0].cpu().numpy()
    pred_f=pred_f[0].cpu().detach().numpy()
    pc1_warp=pc1_warp[0].cpu().detach().numpy()
    gt=gt[0].transpose(0,1).contiguous().cpu().detach().numpy()
    pc1_warp_gt=pc_1+gt
    error = np.linalg.norm(pc1_warp - pc1_warp_gt, axis = 0)
    mask = mask[0].cpu().numpy()

    fig = plt.figure(figsize=(10, 6))

    x_locator = MultipleLocator(10)
    y_locator = MultipleLocator(10)

    x_flow, y_flow = pred_f[0], pred_f[1]
    rad = np.sqrt(np.square(x_flow) + np.square(y_flow))
    x_gt, y_gt = gt[0], gt[1]
    
    rad_max = np.max(rad)
    epsilon = 1e-5
    x_flow = x_flow / (rad_max + epsilon)
    y_flow = y_flow / (rad_max + epsilon)
 
    x_gt = x_gt / (rad_max + epsilon)
    y_gt = y_gt / (rad_max + epsilon)

    yy = np.linspace(-12.5, 12.5, 1000)
    yy1 = np.linspace(-10, 10, 1000)
    xx1 = np.sqrt(10**2-yy1**2)
    xx2 = np.sqrt(20**2-yy**2)
    xx3 = np.sqrt(30**2-yy**2)
    xx4 = np.sqrt(40**2-yy**2)
    xx5 = np.sqrt(50**2-yy**2)

    xx = np.linspace(0, 60, 1000)
    yy2 = np.zeros(xx.shape)
    yy3 = xx * np.tan(5*np.pi/180)
    yy4 = xx * np.tan(-5*np.pi/180)
    yy5 = xx * np.tan(10*np.pi/180)
    yy6 = xx * np.tan(-10*np.pi/180)
    yy7 = xx * np.tan(15*np.pi/180)
    yy8 = xx * np.tan(-15*np.pi/180)
 
    ax1 = plt.gca()
    
    colors = flow_xy_to_colors(x_flow, -y_flow)

    ax1.scatter(pc_1[0], pc_1[1], c = colors/255, marker='o', s=6)
    
    ax1.plot(xx1, yy1, linewidth=0.5, color='white')
    ax1.plot(xx2, yy, linewidth=0.5, color='white')
    ax1.plot(xx3, yy, linewidth=0.5, color='white')
    ax1.plot(xx4, yy, linewidth=0.5, color='white')
    ax1.plot(xx5, yy, linewidth=0.5, color='white')
    ax1.plot(xx, yy2, linewidth=0.5, color='white')
    ax1.plot(xx, yy3, linewidth=0.5, color='white')
    ax1.plot(xx, yy4, linewidth=0.5, color='white')
    ax1.plot(xx, yy5, linewidth=0.5, color='white')
    ax1.plot(xx, yy6, linewidth=0.5, color='white')
    ax1.plot(xx, yy7, linewidth=0.5, color='white')
    ax1.plot(xx, yy8, linewidth=0.5, color='white')
  
    ax1.text(10-0.55, -0.3, '10', fontsize=12, ma= 'center', color = 'white')
    ax1.text(20-0.55, -0.3, '20', fontsize=12, ma = 'center', color = 'white')
    ax1.text(30-0.55, -0.3, '30', fontsize=12, ma = 'center', color = 'white')
    ax1.text(40-0.55, -0.3, '40', fontsize=12, ma = 'center', color = 'white')
    ax1.text(50-0.55, -0.3, '50', fontsize=12, ma = 'center', color = 'white')

    ax1.set_xlim([0, 60])
    ax1.set_ylim([-15, 15])
    ax1.set_box_aspect(0.5)

    ax1.patch.set_facecolor(np.array([80, 80, 80])/255)                 
  
    [ax1.spines[loc_axis].set_visible(False) for loc_axis in ['top','right','bottom','left']]
    ax1.set_xticks([])
    ax1.set_yticks([])

    fig.tight_layout()
    path_im=args.vis_path_flow+'/'+'{}.png'.format(num_pcs)
    fig.savefig(path_im, dpi=200)
    fig.clf
    plt.cla
    plt.close('all')



def visulize_result_2D_seg_pre(pc1, pc2, mask, pred_m, num_pcs, args):


    pc_1=pc1[0].cpu().numpy()
    pc_2=pc2[0].cpu().numpy()

    mask = mask[0].cpu().numpy()
    pred_m = pred_m[0].cpu().numpy()

    fig = plt.figure(figsize=(10, 6))

    x_locator = MultipleLocator(10)
    y_locator = MultipleLocator(10)

    yy = np.linspace(-12.5, 12.5, 1000)
    yy1 = np.linspace(-10, 10, 1000)
    xx1 = np.sqrt(10**2-yy1**2)
    xx2 = np.sqrt(20**2-yy**2)
    xx3 = np.sqrt(30**2-yy**2)
    xx4 = np.sqrt(40**2-yy**2)
    xx5 = np.sqrt(50**2-yy**2)

    xx = np.linspace(0, 60, 1000)
    yy2 = np.zeros(xx.shape)
    yy3 = xx * np.tan(5*np.pi/180)
    yy4 = xx * np.tan(-5*np.pi/180)
    yy5 = xx * np.tan(10*np.pi/180)
    yy6 = xx * np.tan(-10*np.pi/180)
    yy7 = xx * np.tan(15*np.pi/180)
    yy8 = xx * np.tan(-15*np.pi/180)
  
    ax1 = plt.gca()


    ax1.scatter(pc_1[0, mask==0],pc_1[1,mask==0], s=6, c=np.array([[255/255, 99/255, 71/255]]))
    ax1.scatter(pc_1[0, mask==1],pc_1[1,mask==1], s=6, c=np.array([[65/255, 105/255, 225/255]]))

    ax1.plot(xx1, yy1, linewidth=0.5, color='white')
    ax1.plot(xx2, yy, linewidth=0.5, color='white')
    ax1.plot(xx3, yy, linewidth=0.5, color='white')
    ax1.plot(xx4, yy, linewidth=0.5, color='white')
    ax1.plot(xx5, yy, linewidth=0.5, color='white')
    ax1.plot(xx, yy2, linewidth=0.5, color='white')
    ax1.plot(xx, yy3, linewidth=0.5, color='white')
    ax1.plot(xx, yy4, linewidth=0.5, color='white')
    ax1.plot(xx, yy5, linewidth=0.5, color='white')
    ax1.plot(xx, yy6, linewidth=0.5, color='white')
    ax1.plot(xx, yy7, linewidth=0.5, color='white')
    ax1.plot(xx, yy8, linewidth=0.5, color='white')
  
    ax1.text(10-0.55, -0.3, '10', fontsize=12, ma= 'center', color = 'white')
    ax1.text(20-0.55, -0.3, '20', fontsize=12, ma = 'center', color = 'white')
    ax1.text(30-0.55, -0.3, '30', fontsize=12, ma = 'center', color = 'white')
    ax1.text(40-0.55, -0.3, '40', fontsize=12, ma = 'center', color = 'white')
    ax1.text(50-0.55, -0.3, '50', fontsize=12, ma = 'center', color = 'white')

    ax1.set_xlim([0, 60])
    ax1.set_ylim([-15, 15])
    ax1.set_box_aspect(0.5)

    ax1.patch.set_facecolor(np.array([80, 80, 80])/255)                 
    
    [ax1.spines[loc_axis].set_visible(False) for loc_axis in ['top','right','bottom','left']]
    ax1.set_xticks([])
    ax1.set_yticks([])

    fig.tight_layout()
    path_im=args.vis_path_seg+'/'+'{}.png'.format(num_pcs)
    fig.savefig(path_im, dpi=200)
    fig.clf
    plt.cla
    plt.close('all')
