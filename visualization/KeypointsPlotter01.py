# Plots the SMPL-X joints on top of image

import datetime
import time
import traceback

from pathlib import Path

import numpy
import smplx
from matplotlib import pyplot
from smplx import SMPLXLayer
from smplx.utils import SMPLXOutput

import os

import torch
import trimesh
import pyrender
import numpy as np
import colorsys
import cv2
import skimage.io
from scipy.spatial.transform import Rotation

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def perspective_projection(points, rotation, translation, intrinsics):
    K = intrinsics
    points = numpy.einsum('ij,kj->ki', rotation, points)
    points = points + translation[None]
    projected_points = points / points[:, 2:]
    projected_points = numpy.einsum('ij,kj->ki', K, projected_points)
    projected_points_2d = projected_points[:, :-1]
    return projected_points_2d


def get_camera_intrinsics(focal_length, resolution, camera_center = None):
    h, w = resolution
    if camera_center is not None:
        cx, cy = camera_center
    else:
        cx, cy = w / 2., h / 2.
    intrinsics = numpy.eye(3)
    intrinsics[0, 0] = focal_length
    intrinsics[1, 1] = focal_length
    intrinsics[0, 2] = cx
    intrinsics[1, 2] = cy
    return intrinsics



def get_rotations_as_axis_angles(rotation_matrices: numpy.ndarray):
    rotation_axis_angles = Rotation.from_matrix(rotation_matrices).as_rotvec()
    # thetas = []
    # for rot_mat in pose_matrices:
    #     theta = rotation_matrix_to_axis_angle(rot_mat)
    #     thetas.append(theta)
    # thetas = np.array(thetas)
    return rotation_axis_angles


def rotation_matrix_to_axis_angle(rot_mat):
    """Converts a rotation matrix to axis-angle representation."""
    theta = cv2.Rodrigues(rot_mat)[0]  # OpenCV's Rodrigues function does this conversion
    return theta.flatten()  # Flatten to a 1D vector


def read_image(image_filepath: Path):
    image = skimage.io.imread(image_filepath)
    return image


def save_image(image_filepath: Path, image: numpy.ndarray):
    skimage.io.imsave(image_filepath, image)
    return


def read_smplx_data(smplx_data_path: Path):
    # Load torch tensors on CPU directly: https://stackoverflow.com/a/78399538/3337089
    torch.serialization.register_package(0, lambda x: x.device.type, lambda x, _: x.cpu())
    with numpy.load(smplx_data_path.as_posix(), allow_pickle=True) as smplx_data:
        smplx_data = {key: smplx_data[key] for key in smplx_data.files}
    return smplx_data


def main():
    test_num = 1
    video_name = 'IMG_0014'
    frame_num = 0
    show_keypoint_index = True

    image_path = Path(f'../../../../../../databases/spree_internal/data/rgb_png/{video_name}/{frame_num:04}.png')
    smplx_data_path = Path(f'../../runs/testing/test{test_num:04}/{video_name}/frames/{frame_num:04}.npz')
    models_dirpath = Path('../data/body_models/smplx')

    image = read_image(image_path)
    smplx_data = read_smplx_data(smplx_data_path)

    h, w = image.shape[:2]
    focal_length = 5000

    shape_params = smplx_data['betas']  # (10, )
    body_pose_matrices = smplx_data['body_pose']  # (21, 3, 3)
    global_rotation_matrix = smplx_data['global_rot']  # (1, 3, 3)
    pose_matrices = numpy.concatenate([global_rotation_matrix, body_pose_matrices], axis=0)  # (22, 3, 3)
    camera_translation = smplx_data['transl']  # (3, )
    camera_center = smplx_data['center']  # (2, )
    num_shape_params = shape_params.shape[0]  # = 10
    camera_intrinsics = get_camera_intrinsics(focal_length, (h, w), camera_center)

    shape_params_tr = torch.from_numpy(shape_params).float().unsqueeze(0)
    pose_matrices_tr = torch.from_numpy(pose_matrices).float().unsqueeze(0)

    # pose_axis_angles = get_rotations_as_axis_angles(pose_matrices)
    # pose_axis_angles_tr = torch.from_numpy(pose_axis_angles).float().unsqueeze(0)
    # smplx_model = smplx.create(models_dirpath.parent.as_posix(), model_type='smplx', num_betas=num_shape_params)
    # smplx_output = smplx_model(betas=shape_params_tr, body_pose=pose_axis_angles_tr[:, 1:], global_orient=pose_axis_angles_tr[:, :1], return_verts=True)

    smplx_model = SMPLXLayer(models_dirpath.as_posix(), num_betas=num_shape_params)
    smplx_output = smplx_model(betas=shape_params_tr, body_pose=pose_matrices_tr[:, 1:], global_orient=pose_matrices_tr[:, :1], pose2rot=False)

    joints3d = smplx_output.joints[0]
    joints2d = perspective_projection(joints3d.cpu().numpy(), rotation=numpy.eye(3),
                                      translation=camera_translation, intrinsics=camera_intrinsics)

    pyplot.imshow(image)
    for joint_index, joint_location in enumerate(joints2d):
        x, y = joint_location
        pyplot.scatter(x, y, c='r', s=8)
        if show_keypoint_index:
            pyplot.text(x, y, str(joint_index), fontsize=8, color='r')
    pyplot.show()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
