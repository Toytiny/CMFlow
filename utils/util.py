import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from lib import pointnet2_utils as pointutils
from scipy.spatial.transform import Rotation as R
# import lib.pointnet2_utils as pointutils

# Batched index_select
def batched_index_select(t, dim, inds):
    dummy = inds.expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out

def project_radar_to_image(pcs, args):

    t_camera_radar = torch.from_numpy(args.t_camera_radar).cuda().float().unsqueeze(0)
    camera_projection_matrix = torch.from_numpy(args.camera_projection_matrix).cuda().float().unsqueeze(0)
    batch_size = pcs.shape[0]
    npoints = pcs.shape[2]
    radar_p = torch.cat((pcs,torch.ones((batch_size,1,npoints)).cuda()),dim=1)
    cam_p = torch.matmul(t_camera_radar, radar_p)
    cam_uvz = torch.matmul(camera_projection_matrix,cam_p)
    cam_u = cam_uvz[:,0]/cam_uvz[:,2]
    cam_v = cam_uvz[:,1]/cam_uvz[:,2]
    cam_uv = torch.cat((cam_u.unsqueeze(2),cam_v.unsqueeze(2)), dim=2)
    return cam_uv


def point_ray_distance(warped_pcs, pixels, args):

    batch_size = warped_pcs.shape[0]
    npoints = warped_pcs.shape[2]

    # Homogeneous pixel coordinate (depth = 1)
    pixels_h = torch.cat((pixels,torch.ones((batch_size,npoints,1)).cuda()),dim=2).transpose(2,1)

    # Transform pixel to camera coordinate frame
    camera_projection_matrix = torch.from_numpy(args.camera_projection_matrix[:3,:3]).float().unsqueeze(0)
    cam_pcs = torch.inverse(camera_projection_matrix).cuda() @ pixels_h
    
    # set camera origin 
    cam_origin = torch.tensor([0,0,0]).cuda().float().unsqueeze(0).unsqueeze(2)
   
    # Find a ray from camera origin to 3D pixels
    vector = cam_pcs - cam_origin
    unit_vector = vector / torch.norm(vector,dim=1).unsqueeze(1)
    
    # Transform warped points from radar to camera coordinates frame
    t_camera_radar = torch.from_numpy(args.t_camera_radar).cuda().float().unsqueeze(0)
    warped_pcs = torch.cat((warped_pcs,torch.ones((batch_size,1,npoints)).cuda()),dim=1)
    warped_pcs_cam = t_camera_radar @ warped_pcs

    # Compute the distance from warped points to rays
    distance = torch.norm(torch.cross(unit_vector,warped_pcs_cam[:,:3]),dim=1)

    return distance
    
    

def rigid_transform_torch(A, B):

    assert A.size() == B.size()

    batch_size, num_rows, num_cols = A.size()
   
    # find mean column wise
    centroid_A = torch.mean(A.transpose(2,1).contiguous(), axis=1)
    centroid_B = torch.mean(B.transpose(2,1).contiguous(), axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(batch_size,num_rows,1)
    centroid_B = centroid_B.reshape(batch_size,num_rows,1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = torch.matmul(Am, Bm.transpose(2,1).contiguous())

    # find rotation
    U, S, V = torch.svd(H)
    Z = torch.matmul(V,U.transpose(2,1).contiguous())
    # special reflection case
    #d= (torch.linalg.det(Z) < 0).type(torch.int8)
    d = torch.zeros(batch_size).type(torch.int8).cuda()
    # -1/1 
    d=d*2-1
    Vc = V.clone()
    Vc[:,2,:]*=-d.view(batch_size,1)
    R = torch.matmul(Vc,U.transpose(2,1).contiguous())
   
    t = torch.matmul(-R, centroid_A)+centroid_B
    
    Trans=torch.cat((torch.cat((R,t),axis=2), torch.tensor([0,0,0,1]).repeat(batch_size,1).cuda().view(batch_size,1,4)),axis=1)

    return Trans

def rigid_transform_3D(A, B):
    
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")
    
    
    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        #print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B
    
    Trans=np.concatenate((np.concatenate((R,t),axis=1), np.expand_dims(np.array([0,0,0,1]),0)),axis=0)

    return Trans


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    dist = torch.maximum(dist,torch.zeros(dist.size()).cuda())
    return dist

def compute_density_loss(xyz1, xyz2, bandwidth):
    '''
    xyz: input points position data, [B, N, C]
    '''
    #import ipdb; ipdb.set_trace()
    B, N, C = xyz1.shape
    sqrdists = square_distance(xyz1, xyz2)
    gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim = -1)

    return xyz_density

def rigid_to_flow(pc,trans):
    
    h_pc = torch.cat((pc,torch.ones((pc.size()[0],1,pc.size()[2])).cuda()),dim=1)
    sf = torch.matmul(trans,h_pc)[:,:3] - pc

    return sf

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def get_matrix_from_ext(ext):
    

    N = np.size(ext,0)
    if ext.ndim==2:
        rot = R.from_euler('ZYX', ext[:,3:], degrees=True)
        rot_m = rot.as_matrix()
        tr = np.zeros((N,4,4))
        tr[:,:3,:3] = rot_m
        tr[:,:3, 3] = ext[:,:3]
        tr[:, 3, 3] = 1
    if ext.ndim==1:
        rot = R.from_euler('ZYX', ext[3:], degrees=True)
        rot_m = rot.as_matrix()
        tr = np.zeros((4,4))
        tr[:3,:3] = rot_m
        tr[:3, 3] = ext[:3]
        tr[ 3, 3] = 1
    return tr

def homogeneous_transformation(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    This function applies the homogenous transform using the dot product.
    :param points: Points to be transformed in a Nx4 numpy array.
    :param transform: 4x4 transformation matrix in a numpy array.
    :return: Transformed points of shape Nx4 in a numpy array.
    """
    if transform.shape != (4, 4):
        raise ValueError(f"{transform.shape} must be 4x4!")
    if points.shape[1] != 4:
        raise ValueError(f"{points.shape[1]} must be Nx4!")
    return transform.dot(points.T).T


def project_3d_to_2d(points: np.ndarray, projection_matrix: np.ndarray):
    """
    This function projects the input 3d ndarray to a 2d ndarray, given a projection matrix.
    :param points: Homogenous points to be projected.
    :param projection_matrix: 4x4 projection matrix.
    :return: 2d ndarray of the projected points.
    """
    if points.shape[-1] != 4:
        raise ValueError(f"{points.shape[-1]} must be 4!")

    uvw = projection_matrix.dot(points.T)
    uvw /= uvw[2]
    uvs = uvw[:2].T
    uvs = np.round(uvs).astype(np.int)

    return uvs