import gc
import argparse
import sys
from pandas import interval_range
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
from utils import *
from utils.model_utils import *
import torch.nn.functional as F
from torch.nn import Module

class SoftChamferLoss(Module):

    def __init__(self, zeta=0.005):
        super(SoftChamferLoss, self).__init__()
        self.zeta = zeta

    def forward(self, pc1, pc2, pc1_warp):

        """ Compute soft chamfer loss (self-supervised)

        Input (torch.Tensor):
            pc1: [B,3,N]
            pc2: [B,3,N]
            pc1_warp: [B,3,N]
        Return (torch.Tensor):
            loss: [1]
        """

        pc1 = pc1.permute(0, 2, 1)
        pc1_warp = pc1_warp.permute(0,2,1)
        pc2 = pc2.permute(0, 2, 1)

        ## use the kernel density estimation to obatin perpoint density
        dens12 = compute_density_loss(pc1, pc2, 1)
        dens21 = compute_density_loss(pc2, pc1, 1)
        ## estimate inlier mask for two point cloud
        mask1 = (dens12>self.zeta).type(torch.int32)
        mask2 = (dens21>self.zeta).type(torch.int32)
        ## square distance mapping between pc1_warp and pc2
        sqrdist12w = square_distance(pc1_warp, pc2) # B N M
        
        dist1_w, _ = torch.topk(sqrdist12w, 1, dim = -1, largest=False, sorted=False)
        dist2_w, _ = torch.topk(sqrdist12w, 1, dim = 1, largest=False, sorted=False)
        dist1_w = dist1_w.squeeze(2)
        dist2_w = dist2_w.squeeze(1)
        dist1_w = F.relu(dist1_w-0.01)
        dist2_w = F.relu(dist2_w-0.01)
        dist1_w = dist1_w * mask1 
        dist2_w = dist2_w * mask2 
        loss =  torch.mean(dist1_w) + torch.mean(dist2_w)

        return loss

class SpatialSmoothnessLoss(Module):

    def __init__(self, alpha=0.5, num_nb=8):
        super(SpatialSmoothnessLoss, self).__init__()
        self.alpha = alpha
        self.num_nb = num_nb

    def forward(self, pc1, pred_flow):
        
        """ Compute spatial smoothness loss (self-supervised)

        Input (torch.Tensor):
            pc1: [B,3,N]
            pred_flow: [B,3,N]
        Return (torch.Tensor):
            loss: [1]
        """

        B = pc1.size()[0] 
        N = pc1.size()[2]
        pc1 = pc1.permute(0, 2, 1)
        npoints = pc1.size(1)
        pred_flow = pred_flow.permute(0, 2, 1)
        sqrdist = square_distance(pc1, pc1) # B N N

        ## compute the neighbour distances in the point cloud
        dists, kidx = torch.topk(sqrdist, self.num_nb+1, dim = -1, largest=False, sorted=True)
        dists = dists[:,:,1:]
        kidx = kidx[:,:,1:]
        dists = torch.maximum(dists,torch.zeros(dists.size()).cuda())
        ## compute the weights according to the distances
        weights = torch.softmax(torch.exp(-dists/self.alpha).view(B,N*self.num_nb),dim=1)
        weights = weights.view(B,N,self.num_nb)
    
        grouped_flow = index_points_group(pred_flow, kidx) 
        diff_flow = (npoints*weights*torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3)).sum(dim = 2) 
        loss =  torch.mean(diff_flow) 

        return loss

class RadialDisplacementLoss(Module):

    def __init__(self, intervel=0.1):
        super(RadialDisplacementLoss, self).__init__()
        self.interval = 0.1
    
    def forward(self, pc1, pred_f, vel1):
        
        """ Compute radial displacement loss (self-supervised)

        Input (torch.Tensor):
            pc1: [B,3,N]
            pred_f: [B,3,N]
            vel1: [B,N], relative radial velocity measurements from radar
        Return (torch.Tensor):
            loss: [1]
        """
        ## the projection of the estimated flow on radical direction
        pred_fr=torch.sum(pred_f*pc1,dim=1)/(torch.norm(pc1,dim=1))
        diff_vel=torch.abs(vel1*self.interval-pred_fr)
        loss= torch.mean(diff_vel)

        return loss
    
class SelfSupervisedLoss(Module):

    def __init__(self, w_sc=1, w_ss=1, w_rd=1):
        super(SelfSupervisedLoss, self).__init__()
        self.w_sc = w_sc
        self.w_ss = w_ss
        self.w_rd = w_rd
        self.sc_loss = SoftChamferLoss()
        self.ss_loss = SpatialSmoothnessLoss()
        self.rd_loss = RadialDisplacementLoss()

    def forward(self, pc1, pc2, pred_f, vel1):
        """ Compute self-supervised losses

        Input (torch.Tensor):
            pc1: [B,3,N]
            pc1: [B,3,N]
            pred_f: [B,3,N]
            vel1: [B,N], relative radial velocity measurements from radar
        Return (torch.Tensor):
            total_loss: [1]
            items: dict
        """
        pc1_warp = pc1 + pred_f
        scloss = self.sc_loss(pc1, pc2, pc1_warp)
        ssloss = self.ss_loss(pc1, pred_f)
        rdloss = self.rd_loss(pc1, pred_f, vel1) 
        total_loss = self.w_sc * scloss + self.w_ss * ssloss + self.w_rd * rdloss
        #total_loss = self.w_ss * ssloss + self.w_sc * scloss
        #total_loss = self.w_rd * rdloss
        #total_loss = torch.zeros(1).cuda()
        items={
            'Loss': total_loss.item(),
            'smoothnessLoss': ssloss.item(),
            'chamferLoss': scloss.item(),
            'veloLoss': rdloss.item(),
        }
        return total_loss, items

class EgoMotionLoss(Module):

    def __init__(self):
        super(EgoMotionLoss, self).__init__()

    def forward(self, pc1, pre_trans, gt_trans):
        """ Compute ego-motion losses

        Input (torch.Tensor):
            pc1: [B,3,N]
            pred_trans: [B,4,4]
            gt_trans: [B,4,4]
        Return (torch.Tensor):
            total_loss: [1]
        """
        pc1_pre = torch.matmul(pre_trans[:,:3,:3], pc1)+pre_trans[:,:3,3].unsqueeze(2)
        pc1_gt = torch.matmul(gt_trans[:,:3,:3], pc1)+gt_trans[:,:3,3].unsqueeze(2)
        loss = torch.mean(torch.norm(pc1_pre-pc1_gt,dim=1))

        return loss

class MotionSegLoss(Module):

    def __init__(self):
        super(MotionSegLoss, self).__init__()
        self.BCEloss= torch.nn.BCELoss()
        #self.focalloss = WeightedFocalLoss()

    def forward(self, mseg_pre, mseg_gt):
        """ Compute motion segmentation losses

        Input (torch.Tensor):
            mseg_pre: [B,N], predicted motion segmentation logits
            mseg_gt: [B,N], ground truth motion segmentation labels
        Return (torch.Tensor):
            total_loss: [1]
        """
        loss = 0
        loss += self.BCEloss(mseg_pre.squeeze(1)[mseg_gt==0], mseg_gt[mseg_gt==0])
        loss += self.BCEloss(mseg_pre.squeeze(1)[mseg_gt==1], mseg_gt[mseg_gt==1])
        loss = loss/2
        #loss = self.loss_obj(mseg_pre.squeeze(1), mseg_gt)
        return loss

class OpticalFlowLoss(Module):

    def __init__(self):
        super(OpticalFlowLoss, self).__init__()
        self.lower_bound = 0.25
    def forward(self, opt, radar_u, radar_v, pc1_warp, mseg_gt, args):
        """ Compute optical flow losses

        Input (torch.Tensor):
            opt: [B,N,2], pseudo optical flow label
            radar_u, radar_v: [B,N], projected radar points on image plane
            pc1_warp: [B,3,N]
            args: Object of ArgParser
        Return (torch.Tensor):
            total_loss: [1]
        """
        N = opt.shape[1]
        # measure the distance from warped 3D points to camera rays
        end_pixels = torch.cat((radar_u.unsqueeze(2), radar_v.unsqueeze(2)),dim=2) + opt
        opt_div = point_ray_distance(pc1_warp, end_pixels, args)
        # set the upper bound to mitigate the effect of noisy labels
        # opt_div = torch.minimum(opt_div, torch.tensor(self.upper_bound).cuda())
        # discard distances lower than lower bound 
        opt_div = F.relu(opt_div-self.lower_bound)
        # find the min distance between target radar points and each end pixel on the image plane
        # pc2_proj = project_radar_to_image(pc2, args)
        # distances = torch.norm(pc2_proj.unsqueeze(1) - end_pixels.unsqueeze(2), dim=3)
        # diste_2, _ = torch.topk(distances, 1, dim = -1, largest=False, sorted=False)
        #end_pixels_pre = project_radar_to_image(pc1_warp, args)
        #opt_div = torch.norm(end_pixels - end_pixels_pre, p=1, dim=2)
        #mask = (opt_div.detach()<self.max_div)
        #opt_div_topk, div_topk_idx = torch.topk(opt_div, np.int(self.use_ratio*N), dim = -1, largest=False, sorted=True)
        mseg_gt = mseg_gt.type(torch.float32).detach()
        loss = torch.sum((1-mseg_gt) * opt_div)/torch.maximum(torch.sum(1-mseg_gt), torch.tensor(1).cuda())

        return loss

class DynamicFlowLoss(Module):
    def __init__(self):
        super(DynamicFlowLoss, self).__init__()
    def forward(self, pred_f, gt_f, dyn_mask):
        """ Compute dynamic flow losses

        Input (torch.Tensor):
            pred_f: [B,3,N]
            gt_f: [B,3,N]
            dyn_mask: [B,N]
        Return (torch.Tensor):
            total_loss: [1]
        """
        loss = torch.sum((1-dyn_mask)*torch.norm(gt_f-pred_f,dim=1))/torch.maximum(torch.sum(1-dyn_mask), torch.tensor(1).cuda())
        return loss

class RadarFlowLoss(Module):

    def __init__(self, w_self=1, w_em=1, w_ms=1, w_opt=0.1, w_dyn=1):
        super(RadarFlowLoss, self).__init__()
        self.w_self, self.w_em, self.w_ms, self.w_opt, self.w_dyn = w_self, w_em, w_ms, w_opt, w_dyn
        self.self_sup_loss = SelfSupervisedLoss()
        self.ego_motion_loss = EgoMotionLoss()
        self.motion_seg_loss = MotionSegLoss()
        self.opt_flow_loss = OpticalFlowLoss()
        self.dyn_flow_loss = DynamicFlowLoss()
        
    def forward(self, args, pc1, pc2, pred_f, vel1, gt_f=None, pre_trans=None, mseg_pre=None, gt_trans=None, \
                             mseg_gt=None, dyn_mask=None, radar_u=None, radar_v=None, opt=None, basic_f=None):

        if args.model == 'raflow':
            self_sup_loss, items = self.self_sup_loss(pc1, pc2, pred_f, vel1)
            total_loss = self.w_self * self_sup_loss

        if args.model == 'cmflow' or args.model == 'cmflow_t':
            self_sup_loss, items = self.self_sup_loss(pc1, pc2, pred_f, vel1)
            ego_motion_loss = self.ego_motion_loss(pc1, pre_trans, gt_trans)
            motion_seg_loss = self.motion_seg_loss(mseg_pre, mseg_gt)
            dyn_flow_loss = self.dyn_flow_loss(pred_f, gt_f, dyn_mask)
            pc1_warp = pc1 + pred_f
            opt_flow_loss = self.opt_flow_loss(opt, radar_u, radar_v, pc1_warp, mseg_gt, args)
            items['egoLoss'] = ego_motion_loss.item()
            items['maskLoss'] = motion_seg_loss.item()
            items['opticalLoss'] = opt_flow_loss.item()
            items['superviseLoss'] = dyn_flow_loss.item()
            total_loss = self.w_self * self_sup_loss + self.w_em * ego_motion_loss +\
                         self.w_ms * motion_seg_loss + self.w_opt * opt_flow_loss + self.w_dyn * dyn_flow_loss

        return total_loss, items

       




            





