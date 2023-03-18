#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_pose_se3(pose_tst, pose_rot):
    """
        Convert a rotation matrix (or euler angles)
        plus a translation vector into a 4x4 pose
        representation.
        Parameters
        ----------
            pose_tst : np.array
                The (x,y,z) of pose.
            pose_rot : np.array
                 The (theta_x, theta_y, theta_z) of pose.
        Returns
        -------
            np.array (4x4)
                The pose 4x4 matrix.
    """
    if pose_rot.shape == (3, 3):
        rot_mat = pose_rot
    else:
        rot_mat = R.from_euler('xyz', pose_rot).as_dcm()
    tst_vec = pose_tst

    se3 = np.eye(4)
    se3[:3, :3] = rot_mat
    se3[:3, 3] = tst_vec

    return se3

def calculate_rpe_vector(gt, pred):
    """
        Gets a vector of relative errors for all poses.
        Parameters
        ----------
            gt : np.array
                The ground truth of transformation.
            pred : np.array
                The predicted of transformation.
        Returns
        -------
            errors : list
                The list of relative errors.
    """
    errors = []
    for i in range(len(gt)):
        # ground truth
        gt_i = gt[i]

        # predict
        pred_i = pred[i]


        error_i = calc_rpe_pair(gt_i, pred_i)
        errors.append(error_i)
        # errors.append(abs(so3_log(error_i[:3, :3])) * 180 / np.pi)

    return errors

def relative_se3(pose_1, pose_2):
    """
        Relative pose between two poses (drift).
        Parameters
        ----------
            pose_1 : np.array
                The first pose.
            pose_2 : np.array
                 The second pose.
        Returns
        -------
            np.float32
                The relative transformation
                pose_1^{⁻1} * pose_2.
    """
    return np.dot(se3_inverse(pose_1), pose_2)

def se3_inverse(pose):
    """
        The inverse of a pose.
        Parameters
        ----------
            pose : np.array
                The pose.
        Returns
        -------
            np.float32
                The inverted pose.
    """
    r_inv = pose[:3, :3].transpose()
    t_inv = -r_inv.dot(pose[:3, 3])

    return convert_pose_se3(t_inv, r_inv)

def calc_rpe_pair(q_i, p_i):
    """
        The relative error between GT and Predict.
        Parameters
        ----------
            q_i : np.array
                The groundtruth transformation a time i.
           
            p_i : np.array
                The predicted transformation at time i.
            
        Returns
        -------
            np.float32
                The relative distance between two poses.
    """

    # get the relative error between then
    error = relative_se3(q_i, p_i)

    return error

def calc_rpe_error(error_vector, error_type='rotation_angle_deg'):
    """
        Calculate an specific error from relatives errors.
        Parameters
        ----------
            error_vector : list
                List of relative errors.
            error_type : str
                Type of relative error to compute.
        Returns
        -------
            error : list
                The error asked by user.
    """
    if error_type == 'translation_part':
        error = [np.linalg.norm(error_i[:3, 3]) for error_i in error_vector]
    elif error_type == 'rotation_part':
        error = [np.linalg.norm(error_i[:3, :3] - np.eye(3)) for error_i in error_vector]
    elif error_type == 'rotation_angle_deg':
        error = [abs(so3_log(error_i[:3, :3])) * 180 / np.pi for error_i in error_vector]
    else:
        raise NotImplementedError

    return error

def so3_log(rot_matrix):
    """
        Gets the rotation vector from
        rotation matrix.
        Parameters
        ----------
            rot_matrix : np.array
                The rotation matrix.
        Returns
        -------
            np.float32
                The error angle.
    """
    rotation_vector = R.from_matrix(rot_matrix).as_rotvec()
    angle = np.linalg.norm(rotation_vector)

    return angle
    
def get_statistics(rpe_vector):
    """
        Statistics of a vector.
        Parameters
        ----------
            rpe_vector : list
                List of errors.
        Returns
        -------
            dict
                Dict with statistics of a list.
    """
    return {
        'max': np.max(rpe_vector),
        'mean': np.mean(rpe_vector),
        'median': np.median(rpe_vector),
        'min': np.min(rpe_vector),
        'rmse': np.sqrt(np.mean(np.power(rpe_vector, 2))),
        'sse': np.sum(np.power(rpe_vector, 2)),
        'std': np.std(rpe_vector),
    }