import os
import argparse
import sys
import copy
import torch
from time import clock
from tqdm import tqdm
import cv2
import open3d as o3d
import numpy as np
from utils import *
from models import *
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from losses import *
from main_util import *
from utils.vis_util import *


def train_one_epoch_seq(args, net, train_loader, opt):

    total_loss = 0
    num_examples = 0
    mode = 'train'
    net.train()
    loss_items =  copy.deepcopy(loss_dict[args.model])
    seq_len = train_loader.dataset.mini_clip_len

    for i, data in tqdm(enumerate(train_loader), total = len(train_loader)):
        # use sequence data in order
        iter_loss = 0
        iter_items = copy.deepcopy(loss_dict[args.model])
        num_examples += args.batch_size
        for j in range(0,seq_len):
            ## reading data from dataloader and transform their format
            pc1, pc2, ft1, ft2, gt_trans, flow_label, \
                fg_mask, interval, radar_u, radar_v, opt_flow = extract_data_info_clip(data, j)
        
            batch_size = pc1.size(0)
    
            vel1 = ft1[:,0]
           
            if args.model == 'cmflow_t':
                
                dyn_mask = extract_dynamic_from_fg(fg_mask,pc1,gt_trans,flow_label.transpose(2,1))
                mseg_gt, _ = mseg_label_RRV(pc1, gt_trans, vel1, interval, args)
                # aggregate pseudo label generated w.r.t rrv and pseudo label 
                mseg_gt[torch.logical_not(dyn_mask==1)]= dyn_mask[torch.logical_not(dyn_mask==1)]
            
                # forward and loss computation
                if j==0:
                    pred_f, mseg_pre, pre_trans, _, gfeat = net(pc1, pc2, ft1, ft2, mseg_gt, mode, None)
                else: 
                    gfeat = gfeat.detach()
                    pred_f, mseg_pre, pre_trans, _, gfeat = net(pc1, pc2, ft1, ft2, mseg_gt, mode, gfeat)
                    
                loss_obj = RadarFlowLoss()
                loss, items = loss_obj(args, pc1, pc2, pred_f, vel1, flow_label.transpose(2,1), pre_trans, mseg_pre, gt_trans,\
                                        mseg_gt, dyn_mask, radar_u, radar_v, opt_flow)
            opt.zero_grad() 
            loss.backward()
            opt.step()

            iter_loss += loss
            for k in iter_items:
                iter_items[k].append(items[k])
        
        iter_loss = iter_loss/seq_len
        for l in iter_items:
            loss_items[l].append(np.mean(np.array(iter_items[l])))
        total_loss += iter_loss.item() * batch_size

        
    total_loss=total_loss/num_examples
    for l in loss_items:
        loss_items[l]=np.mean(np.array(loss_items[l]))
    
    return total_loss, loss_items


def extract_data_info_clip(seq_data, idx):

    pc1, pc2, ft1, ft2, trans, gt, mask, interval, radar_u, radar_v, opt_flow = seq_data
    pc1 = pc1[:,idx].cuda().transpose(2,1).contiguous()
    pc2 = pc2[:,idx].cuda().transpose(2,1).contiguous()
    ft1 = ft1[:,idx].cuda().transpose(2,1).contiguous()
    ft2 = ft2[:,idx].cuda().transpose(2,1).contiguous()
    radar_v = radar_v[:,idx].cuda().float()
    radar_u = radar_u[:,idx].cuda().float()
    opt_flow = opt_flow[:,idx].cuda().float()
    mask = mask[:,idx].cuda().float()
    trans = trans[:,idx].cuda().float()
    interval = interval[:,idx].cuda().float()
    gt = gt[:,idx].cuda().float()

    return pc1, pc2, ft1, ft2, trans, gt, mask, interval, radar_u, radar_v, opt_flow


def eval_one_epoch_seq(args, net, eval_loader, textio):

    
    net.eval()
    
    num_pcs=0 
    
    sf_metric = {'rne':0, '50-50 rne': 0, 'mov_rne': 0, 'stat_rne': 0,\
                 'sas': 0, 'ras': 0, 'epe': 0, 'accs': 0, 'accr': 0}

    seg_metric = {'acc': 0, 'miou': 0, 'sen': 0}
    pose_metric = {'RRE': 0, 'RTE': 0}


    seq_len = eval_loader.dataset.mini_clip_len    
    batch_size = eval_loader.batch_size
    gt_trans_all = torch.zeros((len(eval_loader)*batch_size*seq_len,4,4)).cuda()
    pre_trans_all = torch.zeros((len(eval_loader)*batch_size*seq_len,4,4)).cuda()

    # start point for inference
    start_point = time.time()

    with torch.no_grad():
        # read sequence data
        for i, data in tqdm(enumerate(eval_loader), total = len(eval_loader)):
            # use sequence data in order
            for j in range(0,seq_len):
                ## reading data from dataloader and transform their format
                pc1, pc2, ft1, ft2, trans, gt, \
                    mask, interval, radar_u, radar_v, opt_flow = extract_data_info_clip(data, j)
        
                if args.model in ['cmflow_t']:
                    if j==0:
                        pred_f, _, pred_t, pred_m, gfeat = net(pc1, pc2, ft1, ft2, None, 'test', None)
                    else:
                        pred_f, _, pred_t, pred_m, gfeat = net(pc1, pc2, ft1, ft2, None, 'test', gfeat)
                
                batch_size = pc1.shape[0]
                ## use estimated scene to warp point cloud 1 
                pc1_warp=pc1+pred_f
                
                ## evaluate the estimated results using ground truth
                batch_res = eval_scene_flow(pc1, pred_f.transpose(2,1).contiguous(), gt, mask, args)
                for metric in sf_metric:
                    sf_metric[metric] += batch_size * batch_res[metric]
                

                ## evaluate the foreground segmentation precision and recall
                if args.model in ['cmflow_t']:
                    seg_res = eval_motion_seg(pred_m, mask)
                    for metric in seg_res:
                        seg_metric[metric] += batch_size * seg_res[metric]
                
                ## Use scene flow correspondence to estimate rigid 3D transformation
                if pred_t is not None:
                    pred_trans = pred_t
                else:
                    pred_trans = rigid_transform_torch(pc1, pc1_warp)
                
                gt_trans_all[num_pcs:num_pcs+batch_size] = trans
                pre_trans_all[num_pcs:num_pcs+batch_size] = pred_trans   

                pose_res = eval_trans_RPE(trans, pred_trans)
                for metric in pose_res:
                    pose_metric[metric] += batch_size * pose_res[metric]
                
                num_pcs+=batch_size

    # end point for inference
    infer_time = time.time()-start_point

    for metric in sf_metric:
        sf_metric[metric] = sf_metric[metric]/num_pcs
    for metric in seg_metric:
        seg_metric[metric] = seg_metric[metric]/num_pcs
    for metric in pose_metric:
        pose_metric[metric] = pose_metric[metric]/num_pcs

    textio.cprint('###The inference speed is %.3fms per frame###'%(infer_time*1000/num_pcs))
    return sf_metric, seg_metric, pose_metric, gt_trans_all, pre_trans_all



def test_one_epoch_seq(args, net, test_loader, textio):

    
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

    gt_trans_all = torch.zeros((len(test_loader),4,4)).cuda()
    pre_trans_all = torch.zeros((len(test_loader),4,4)).cuda()

    # start point for inference
    start_point = time()

    with torch.no_grad():
        clips_info = test_loader.dataset.clips_info
        clips_name = []
        clips_st_index = []
        # extract clip info
        for i in range(len(clips_info)):
           clips_name.append(clips_info[i]['clip_name'])
           clips_st_index.append(clips_info[i]['index'][0])
        # read data in order
        num_clip = 0
        seq_len = test_loader.dataset.update_len
        for i, data in tqdm(enumerate(test_loader), total = len(test_loader)):
            
            ## reading data from dataloader and transform their format
            pc1, pc2, ft1, ft2, trans, gt, \
                mask, interval, radar_u, radar_v, opt_flow = extract_data_info(data)
        
            if args.model in ['cmflow_t']:
                #if i==clips_st_index[num_clip]:
                if i==clips_st_index[num_clip] or i%seq_len==0:
                    pred_f, stat_cls, pred_t, pred_m, gfeat = net(pc1, pc2, ft1, ft2, None, 'test', None)
                    if num_clip<(len(clips_name)-1):
                        num_clip +=1
                else:
                    pred_f, stat_cls, pred_t, pred_m, gfeat = net(pc1, pc2, ft1, ft2, None, 'test', gfeat)

            ## use estimated scene to warp point cloud 1 
            pc1_warp=pc1+pred_f

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
            
            ## evaluate the estimated results using ground truth
            batch_res = eval_scene_flow(pc1, pred_f.transpose(2,1).contiguous(), gt, mask, args)
            for metric in sf_metric:
                sf_metric[metric] += batch_res[metric]

            ## evaluate the foreground segmentation precision and recall
            if args.model in ['cmflow_t']:
                seg_res = eval_motion_seg(pred_m, mask)
                for metric in seg_res:
                    seg_metric[metric] += seg_res[metric]
                
            ## Use scene flow correspondence to estimate rigid 3D transformation
            if pred_t is not None:
                pred_trans = pred_t
            else:
                pred_trans = rigid_transform_torch(pc1, pc1_warp)
                
            gt_trans_all[num_pcs:num_pcs+1] = trans
            pre_trans_all[num_pcs:num_pcs+1] = pred_trans   

            pose_res = eval_trans_RPE(trans, pred_trans)
            for metric in pose_res:
                pose_metric[metric] += pose_res[metric]
                
            num_pcs+=1

    # end point for inference
    infer_time = time()-start_point

    for metric in sf_metric:
        sf_metric[metric] = sf_metric[metric]/num_pcs
    for metric in seg_metric:
        seg_metric[metric] = seg_metric[metric]/num_pcs
    for metric in pose_metric:
        pose_metric[metric] = pose_metric[metric]/num_pcs

    textio.cprint('###The inference speed is %.3fms per frame###'%(infer_time*1000/num_pcs))
    return sf_metric, seg_metric, pose_metric, gt_trans_all, pre_trans_all