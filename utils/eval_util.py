import numpy as np
from .odometry_util import *

def get_carterian_res(pc, sensor, args):

    ## measure resolution for r/theta/phi
    if sensor == 'radar': # LRR30
        r_res = args.radar_res['r_res']# m
        theta_res = args.radar_res['theta_res'] # radian
        phi_res = args.radar_res['phi_res']  # radian
        
    if sensor == 'lidar': # HDL-64E
        r_res = 0.04 # m
        theta_res = 0.4 * np.pi/180 # radian
        phi_res = 0.08 *np.pi/180  # radian
         
    res = np.array([r_res, theta_res, phi_res])
    ## x y z
    x = pc[:,0]
    y = pc[:,1]
    z = pc[:,2]
    
    ## from xyz to r/theta/phi (range/elevation/azimuth)
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arcsin(z/r)
    phi = np.arctan2(y,x)
    
    ## compute xyz's gradient about r/theta/phi 
    grad_x = np.stack((np.cos(phi)*np.cos(theta), -r*np.sin(theta)*np.cos(phi), -r*np.cos(theta)*np.sin(phi)),axis=2)
    grad_y = np.stack((np.sin(phi)*np.cos(theta), -r*np.sin(phi)*np.sin(theta), r*np.cos(theta)*np.cos(phi)),axis=2)
    grad_z = np.stack((np.sin(theta), r*np.cos(theta), np.zeros((np.size(x,0),np.size(x,1)))),axis=2)
    
    ## measure resolution for xyz (different positions have different resolution)
    x_res = np.sum(abs(grad_x) * res,axis=2)
    y_res = np.sum(abs(grad_y) * res,axis=2)
    z_res = np.sum(abs(grad_z) * res,axis=2)
    
    xyz_res = np.stack((x_res,y_res,z_res),axis=2)
    
    return xyz_res
        
def eval_scene_flow(pc, pred, labels, mask, args):
    
    pc = pc.cpu().numpy()
    pred = pred.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    mask = mask.cpu().numpy()
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)
    error_x = np.abs(pred[0,:,0]-labels[0,:,0])
    error_y = np.abs(pred[0,:,1]-labels[0,:,1])
    error_z = np.abs(pred[0,:,2]-labels[0,:,2])
    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) 
    
    ## compute traditional metric for scene flow
    epe = np.mean(error)

    accs = np.sum(np.logical_or((error <= 0.05), (error/gtflow_len <=0.05)))/(np.size(pred,0)*np.size(pred,1))
    accr = np.sum(np.logical_or((error <= 0.10), (error/gtflow_len <=0.10)))/(np.size(pred,0)*np.size(pred,1))
   
    ## obtain x y z measure resolution for each point (radar lidar)
    xyz_res_r = get_carterian_res(pc, 'radar', args) 
    res_r = np.sqrt(np.sum(xyz_res_r,2)+1e-20)
    xyz_res_l = get_carterian_res(pc, 'lidar', args) 
    res_l = np.sqrt(np.sum(xyz_res_l,2)+1e-20)
    
    ## calcualte Resolution-Normalized Error
    re_error = error/(res_r/res_l)
    rne = np.mean(re_error)
    mov_rne = np.sum(re_error[mask==0])/(np.sum(mask==0)+1e-6)
    stat_rne = np.mean(re_error[mask==1])
    avg_rne = (mov_rne+stat_rne)/2
    
    ## calculate Strict/Relaxed Accuracy Score
    sas = np.sum(np.logical_or((re_error <= 0.10), (re_error/gtflow_len <= 0.10)))/(np.size(pred,0)*np.size(pred,1))
    ras = np.sum(np.logical_or((re_error <= 0.20), (re_error/gtflow_len <= 0.20)))/(np.size(pred,0)*np.size(pred,1))
    
   
    sf_metric = {'rne':rne, '50-50 rne': avg_rne, 'mov_rne': mov_rne, 'stat_rne': stat_rne,\
                 'sas': sas, 'ras': ras, 'epe':epe, 'accs': accs, 'accr': accr}
    
 
    return sf_metric


def eval_trans_RPE(gt_trans,rigid_trans):
    
    ## Use the RPE to evaluate the prediction
    gt_trans = gt_trans.cpu().numpy()
    rigid_trans = rigid_trans.cpu().detach().numpy()
    error_sf=calculate_rpe_vector(gt_trans,rigid_trans)
    trans_error_sf=calc_rpe_error(error_sf, error_type='translation_part')
    rot_error_sf=calc_rpe_error(error_sf, error_type='rotation_part')
    angle_error_sf=calc_rpe_error(error_sf, error_type='rotation_angle_deg')
    pose_metric = {'RTE': np.array(trans_error_sf).mean(),
                    'RAE': np.array(angle_error_sf).mean()}
    
    return pose_metric
    
def eval_motion_seg(pre, gt):
    
    pre = pre.cpu().detach().numpy()
    gt = gt.cpu().numpy()
    tp = np.logical_and((pre==1),(gt==1)).sum()
    tn = np.logical_and((pre==0),(gt==0)).sum()
    fp = np.logical_and((pre==1),(gt==0)).sum()
    fn = np.logical_and((pre==0),(gt==1)).sum()
    acc = (tp+tn)/(tp+tn+fp+fn)
    sen = tp/(tp+fn+1e-10)
    miou = 0.5*(tp/(tp+fp+fn+1e-10)+tn/(tn+fp+fn+1e-10))
    seg_metric = {'acc': acc, 'miou': miou, 'sen': sen}
    
    return seg_metric
