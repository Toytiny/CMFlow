import os
import argparse
import sys
import copy
import torch
import ujson
from time import clock
from tqdm import tqdm
import cv2
import open3d as o3d
import numpy as np
from utils import *
from utils.vis_util import *
from models import *
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from losses import *



def extract_data_info(data):

    pc1, pc2, ft1, ft2, trans, gt, mask, interval, radar_u, radar_v, opt_flow = data
    pc1 = pc1.cuda().transpose(2,1).contiguous()
    pc2 = pc2.cuda().transpose(2,1).contiguous()
    ft1 = ft1.cuda().transpose(2,1).contiguous()
    ft2 = ft2.cuda().transpose(2,1).contiguous()
    radar_v = radar_v.cuda().float()
    radar_u = radar_u.cuda().float()
    opt_flow = opt_flow.cuda().float()
    mask = mask.cuda().float()
    trans = trans.cuda().float()
    interval = interval.cuda().float()
    gt = gt.cuda().float()

    return pc1, pc2, ft1, ft2, trans, gt, mask, interval, radar_u, radar_v, opt_flow


def train_one_epoch(args, net, train_loader, opt):
    
    num_examples = 0
    total_loss = 0
    mode = 'train'
    loss_items =  copy.deepcopy(loss_dict[args.model])

    for i, data in tqdm(enumerate(train_loader), total = len(train_loader)):
        
        ## reading data from dataloader and transform their format
        pc1, pc2, ft1, ft2, gt_trans, flow_label, \
            fg_mask, interval, radar_u, radar_v, opt_flow = extract_data_info(data)
        vel1 = ft1[:,0]
        batch_size = pc1.size(0)
        num_examples += batch_size

        ## feed data into the model and compute loss
        # self-supervised or cross-modal supervised learning
        
        if args.model=='raflow':
            _, pred_f, _,_ = net(pc1, pc2, ft1, ft2, interval)
            loss_obj = RadarFlowLoss()
            loss, items = loss_obj(args, pc1, pc2, pred_f, vel1)
            
        if args.model == 'cmflow':
            dyn_mask = extract_dynamic_from_fg(fg_mask,pc1,gt_trans,flow_label.transpose(2,1))
            mseg_gt, _ = mseg_label_RRV(pc1, gt_trans, vel1, interval, args)
            # aggregate pseudo label generated w.r.t rrv and pseudo label 
            mseg_gt[torch.logical_not(dyn_mask==1)] = dyn_mask[torch.logical_not(dyn_mask==1)]
            # forward and loss computation
            pred_f, mseg_pre, pre_trans, _ = net(pc1, pc2, ft1, ft2, mseg_gt, mode)
            loss_obj = RadarFlowLoss()
            loss, items = loss_obj(args, pc1, pc2, pred_f, vel1, flow_label.transpose(2,1), pre_trans, mseg_pre, gt_trans,\
                                        mseg_gt, dyn_mask, radar_u, radar_v, opt_flow)
        
        opt.zero_grad() 
        loss.backward()
        opt.step()
        
        total_loss += loss.item() * batch_size
        

        for l in loss_items:
            loss_items[l].append(items[l]) 

  
    total_loss=total_loss*1.0/num_examples
    
    for l in loss_items:
        loss_items[l]=np.mean(np.array(loss_items[l]))
    
    return total_loss, loss_items


def eval_one_epoch(args, net, eval_loader, textio):

   
    net.eval()
    
    if args.save_res: 
        args.save_res_path ='checkpoints/'+args.exp_name+"/results/"
        num_seq = 0
        clip_info = args.clips_info[num_seq]
        seq_res_path = os.path.join(args.save_res_path, clip_info['clip_name'])
        if not os.path.exists(seq_res_path):
            os.makedirs(seq_res_path)

    num_pcs=0 
    
    sf_metric = {'rne':0, '50-50 rne': 0, 'mov_rne': 0, 'stat_rne': 0,\
                 'sas': 0, 'ras': 0, 'epe': 0, 'accs': 0, 'accr': 0}

    seg_metric = {'acc': 0, 'miou': 0, 'sen': 0}
    pose_metric = {'RTE': 0, 'RAE': 0}
    
    gt_trans_all = torch.zeros((len(eval_loader)*eval_loader.batch_size,4,4)).cuda()
    pre_trans_all = torch.zeros((len(eval_loader)*eval_loader.batch_size,4,4)).cuda()
    infer_time = 0

    for i, data in tqdm(enumerate(eval_loader), total = len(eval_loader)):

    
        pc1, pc2, ft1, ft2, trans, gt , mask, interval, radar_u, radar_v, padding_opt = data
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        ft1 = ft1.cuda().transpose(2,1).contiguous()
        ft2 = ft2.cuda().transpose(2,1).contiguous()
        mask = mask.cuda()
        interval = interval.cuda().float()
        gt = gt.cuda().float()

        batch_size = pc1.size(0)
        vel1 = ft1[:,0]
        
        with torch.no_grad():
            # start point for inference
            start_point = time()
            
            pred_t = None
            
            if args.model=='raflow':
                _, pred_f, pred_t, pred_m = net(pc1, pc2, ft1, ft2, interval)
            if args.model=='cmflow':
                pred_f, stat_cls, pred_t, pred_m = net(pc1, pc2, ft1, ft2, None, 'test')
        
            # end point for inference
            infer_time += time()-start_point
            # use estimated scene to warp point cloud 1 
            pc1_warp=pc1 + pred_f

            if args.save_res:
                res = {
                    'pc1': pc1[0].cpu().numpy().tolist(),
                    'pc2': pc2[0].cpu().numpy().tolist(),
                    'pred_f': pred_f[0].cpu().detach().numpy().tolist(),
                    'pred_m': pred_m[0].cpu().detach().numpy().astype(float).tolist(),
                    'pred_t': pred_t[0].cpu().detach().numpy().astype(float).tolist(),
                }
                
                if num_pcs < clip_info['index'][1]:
                    res_path = os.path.join(seq_res_path, '{}.json'.format(num_pcs))
                else:
                    num_seq += 1
                    clip_info = args.clips_info[num_seq]
                    seq_res_path = os.path.join(args.save_res_path, clip_info['clip_name'])
                    if not os.path.exists(seq_res_path):
                        os.makedirs(seq_res_path)
                    res_path = os.path.join(seq_res_path, '{}.json'.format(num_pcs))
                
                ujson.dump(res,open(res_path, "w"))

            if args.vis:
                visulize_result_2D_pre(pc1, pc2, pred_f, pc1_warp, gt, num_pcs, mask, args)
                visulize_result_2D_seg_pre(pc1, pc2, mask, pred_m, num_pcs, args)
                
            # evaluate the estimated results using ground truth
            batch_res = eval_scene_flow(pc1, pred_f.transpose(2,1).contiguous(), gt, mask, args)
            for metric in sf_metric:
                sf_metric[metric] += batch_size * batch_res[metric]


            ## evaluate the foreground segmentation precision and recall
            if args.model in ['raflow', 'cmflow']:
                seg_res = eval_motion_seg(pred_m, mask)
                for metric in seg_res:
                    seg_metric[metric] += batch_size * seg_res[metric]
            
            ## evaluate the ego-motion estimation results
            pred_trans = pred_t
            gt_trans_all[num_pcs:(num_pcs+batch_size)] = trans
            pre_trans_all[num_pcs:(num_pcs+batch_size)] = pred_trans   

            pose_res = eval_trans_RPE(trans, pred_trans)
            for metric in pose_res:
                pose_metric[metric] += batch_size * pose_res[metric]
            
            num_pcs+=batch_size

    for metric in sf_metric:
        sf_metric[metric] = sf_metric[metric]/num_pcs
    for metric in seg_metric:
        seg_metric[metric] = seg_metric[metric]/num_pcs
    for metric in pose_metric:
        pose_metric[metric] = pose_metric[metric]/num_pcs

    textio.cprint('###The inference speed is %.3fms per frame###'%(infer_time*1000/num_pcs))

    return sf_metric, seg_metric, pose_metric, gt_trans_all, pre_trans_all


def extract_dynamic_from_fg(mask, pc1, trans, gt):
    
    # get rigid flow labels for all points
    gt_sf_rg = rigid_to_flow(pc1,trans)
    
    gt = gt.transpose(2,1)
    gt_sf_rg = gt_sf_rg.transpose(2,1)
    # get non-rigid components for points
    flow_nr = gt_sf_rg - gt
    
    # obtain the motion segmentation mask 
    fg_mask = (mask!=1)
    mask[torch.norm(flow_nr*fg_mask.unsqueeze(2),dim=2)<0.05]=1
    mask[mask!=1] = 0

    return mask
    
    
def probabilistic_label_opt(pc1, trans, radar_u, radar_v, opt_flow, args):

    batch_size = pc1.size(0)
    npoints = pc1.size(2)
    
    gt_sf_rg = rigid_to_flow(pc1,trans)
    gt_wp_rg = gt_sf_rg + pc1
    end_pixels = torch.cat((radar_u.unsqueeze(2), radar_v.unsqueeze(2)),dim=2) + opt_flow
    rg_proj = project_radar_to_image(gt_wp_rg, args)
    residual = torch.norm(rg_proj - end_pixels, dim=2)
    prob_m = torch.exp(-(residual**2)/(2*args.sigma_opt**2))

    return prob_m


def probabilistic_label_RRV(pc1,trans,vel1,interval,args):
    
    batch_size = pc1.size(0)
    npoints = pc1.size(2)
    gt_sf_rg = rigid_to_flow(pc1,trans)
    gt_sf_rg_proj=torch.sum(gt_sf_rg*pc1,dim=1)/(torch.norm(pc1,dim=1))
    residual=(vel1*interval.unsqueeze(1)-gt_sf_rg_proj)
    prob_m = torch.exp(-(residual**2)/(2*args.sigma_rrv**2))

    return prob_m

def mseg_label_RRV(pc1, trans, vel1, interval, args):

    gt_sf_rg = rigid_to_flow(pc1,trans)
    gt_sf_rg_proj=torch.sum(gt_sf_rg*pc1,dim=1)/(torch.norm(pc1,dim=1))
    residual=abs(vel1-gt_sf_rg_proj/interval.unsqueeze(1))
    N = pc1.shape[2]
    #low_residual, _ = torch.topk(residual, np.int(args.bs_ratio*N), dim=1, largest=False)
    bs_residual = torch.mean(residual, dim=1).unsqueeze(1)
    #bs_residual = 0
    # 1 denotes static, 0 denotes moving
    mseg_label = ((residual-bs_residual)<args.vr_thres).type(torch.float32)

    return mseg_label, residual

def mseg_label_opt(pc1, trans, radar_u, radar_v, opt_flow, args):

    gt_sf_rg = rigid_to_flow(pc1,trans)
    gt_wp_rg = gt_sf_rg + pc1
    end_pixels = torch.cat((radar_u.unsqueeze(2), radar_v.unsqueeze(2)),dim=2) + opt_flow
    rg_proj = project_radar_to_image(gt_wp_rg, args)
    residual = torch.norm(rg_proj - end_pixels, dim=2)

    mseg_label = ((residual)<args.opt_thres).type(torch.float32)
    #prob_m = torch.exp(-(residual**2)/(2*args.sigma_opt**2))

    return mseg_label

def plot_loss_epoch(train_items_iter, args, epoch):
    
    plt.clf()
    plt.plot(np.array(train_items_iter['Loss']).T, 'b')
    plt.plot(np.array(train_items_iter['chamferLoss']).T, 'k')
    plt.plot(np.array(train_items_iter['veloLoss']).T, 'g')
    plt.plot(np.array(train_items_iter['smoothnessLoss']).T, 'c')
    plt.plot(np.array(train_items_iter['egoLoss']).T, 'm')
    plt.plot(np.array(train_items_iter['maskLoss']).T, 'r')
    plt.plot(np.array(train_items_iter['opticalLoss']).T, 'y')
    plt.plot(np.array(train_items_iter['superviseLoss']).T, 'r')
    plt.legend(['Total','chamferLoss','veloLoss','Smoothness','egoLoss', 'maskLoss',\
        'opticalLoss', 'superviseLoss'], loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('checkpoints/%s/loss_train/loss_train_%s.png' %(args.exp_name,epoch),dpi=500)
 
    

