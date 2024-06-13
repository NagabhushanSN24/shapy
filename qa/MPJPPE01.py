# Computes Mean Per Joint Projected Pixel Error between SMPL-X joints and VitPose_Halpe Keypoints

import collections.abc
import datetime
import json
import time
import traceback
from pathlib import Path

import numpy
import pandas
import torch
from smplx import SMPLXLayer
from tqdm import tqdm

this_filepath = Path(__file__)
this_filename = this_filepath.stem
this_qa_name = this_filename
num_round_off_digits = 2

# https://github.com/spree3d/bedlam/blob/main/train/core/smplx_trainer_spreeV2.py#L42--L52
body_mapping = numpy.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                            8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                            63, 64, 65], dtype=numpy.int32)

lhand_mapping = numpy.array([20, 37, 38, 39, 66, 25, 26, 27,
                             67, 28, 29, 30, 68, 34, 35, 36, 69,
                             31, 32, 33, 70], dtype=numpy.int32)

rhand_mapping = numpy.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                             43, 44, 45, 73, 49, 50, 51, 74, 46,
                             47, 48, 75], dtype=numpy.int32)


class MPJPPE:
    def __init__(self, smplx_models_dirpath: Path, num_shape_params: int, resolution: tuple):
        self.smplx_models_dirpath = smplx_models_dirpath
        self.num_shape_params = num_shape_params
        self.resolution = resolution
        self.smplx_model = SMPLXLayer(smplx_models_dirpath.as_posix(), num_betas=num_shape_params)
        return

    def compute_mpjppe(self, smplx_data_path: Path, gt_keypoints_path: Path):
        smplx_data = self.read_smplx_data(smplx_data_path)
        pred_joints2d = self.get_pred_joints2d(smplx_data)

        gt_keypoints_data = self.read_gt_keypoints_data(gt_keypoints_path)
        gt_keypoints, weights = self.get_gt_keypoints(gt_keypoints_data)

        pred_joints2d_selected = self.select_pred_joints2d(pred_joints2d)
        gt_keypoints_selected, weights_selected = self.select_gt_keypoints(gt_keypoints, weights)
        mpjppe = self.compute_weighted_error(pred_joints2d_selected, gt_keypoints_selected, weights_selected)
        return mpjppe

    @staticmethod
    def compute_weighted_error(pred_joints2d, gt_keypoints, weight):
        error = pred_joints2d - gt_keypoints
        distance = numpy.linalg.norm(error, ord=2, axis=1)
        weighted_distance = weight[:, 0] * distance
        mean_weighted_distance = numpy.mean(weighted_distance)
        return mean_weighted_distance

    def get_pred_joints2d(self, smplx_data: dict):
        h, w = self.resolution
        focal_length = 5000

        shape_params = smplx_data['betas']  # (10, )
        body_pose_matrices = smplx_data['body_pose']  # (21, 3, 3)
        global_rotation_matrix = smplx_data['global_rot']  # (1, 3, 3)
        pose_matrices = numpy.concatenate([global_rotation_matrix, body_pose_matrices], axis=0)  # (22, 3, 3)
        camera_translation = smplx_data['transl']  # (3, )
        camera_center = smplx_data['center']  # (2, )
        camera_intrinsics = self.get_camera_intrinsics(focal_length, (h, w), camera_center)

        shape_params_tr = torch.from_numpy(shape_params).float().unsqueeze(0)
        pose_matrices_tr = torch.from_numpy(pose_matrices).float().unsqueeze(0)

        smplx_output = self.smplx_model(betas=shape_params_tr, body_pose=pose_matrices_tr[:, 1:],
                                        global_orient=pose_matrices_tr[:, :1], pose2rot=False)

        joints3d = smplx_output.joints[0]
        joints2d = self.perspective_projection(joints3d.cpu().numpy(), rotation=numpy.eye(3),
                                               translation=camera_translation, intrinsics=camera_intrinsics)
        return joints2d

    @staticmethod
    def perspective_projection(points, rotation, translation, intrinsics):
        K = intrinsics
        points = numpy.einsum('ij,kj->ki', rotation, points)
        points = points + translation[None]
        projected_points = points / points[:, 2:]
        projected_points = numpy.einsum('ij,kj->ki', K, projected_points)
        projected_points_2d = projected_points[:, :-1]
        return projected_points_2d

    @staticmethod
    def get_camera_intrinsics(focal_length, resolution, camera_center=None):
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

    @staticmethod
    def get_gt_keypoints(gt_keypoints_data: dict):
        pose_keypoints2d = numpy.array(gt_keypoints_data['people'][0]['pose_keypoints_2d']).reshape(-1,
                                                                                                    3)  # 25 keypoints
        hand_left_keypoints2d = numpy.array(gt_keypoints_data['people'][0]['hand_left_keypoints_2d']).reshape(-1,
                                                                                                              3)  # 21 keypoints
        hand_right_keypoints2d = numpy.array(gt_keypoints_data['people'][0]['hand_right_keypoints_2d']).reshape(-1,
                                                                                                                3)  # 21 keypoints
        face_keypoints2d = numpy.array(gt_keypoints_data['people'][0]['face_keypoints_2d']).reshape(-1, 3)[:-2][
                           :-17]  # 70 -> 51 keypoints
        gt_keypoints2d = numpy.concatenate(
            [pose_keypoints2d, hand_left_keypoints2d, hand_right_keypoints2d, face_keypoints2d], axis=0)
        gt_keypoints2d, weights = gt_keypoints2d[:, :2], gt_keypoints2d[:, 2:]
        return gt_keypoints2d, weights

    @staticmethod
    def select_pred_joints2d(pred_joints2d: numpy.ndarray):
        body_keypoints2d = pred_joints2d[body_mapping]  # 25 keypoints
        lhand_keypoints2d = pred_joints2d[lhand_mapping]  # 21 keypoints
        rhand_keypoints2d = pred_joints2d[rhand_mapping]  # 21 keypoints
        # face_keypoints2d = pred_joints2d[face_mapping]
        pred_joints2d_selected = numpy.concatenate([body_keypoints2d, lhand_keypoints2d, rhand_keypoints2d], axis=0)
        return pred_joints2d_selected

    def select_gt_keypoints(self, gt_keypoints: numpy.ndarray, weights: numpy.ndarray):
        num_required_keypoints = body_mapping.size + lhand_mapping.size + rhand_mapping.size
        gt_keypoints_selected = gt_keypoints[:num_required_keypoints]
        weights_selected = weights[:num_required_keypoints]
        return gt_keypoints_selected, weights_selected

    @staticmethod
    def read_smplx_data(smplx_data_path: Path):
        # Load torch tensors on CPU directly: https://stackoverflow.com/a/78399538/3337089
        # torch.serialization.register_package(0, lambda x: x.device.type, lambda x, _: x.cpu())
        smplx_dict = {}
        with numpy.load(smplx_data_path.as_posix(), allow_pickle=True) as smplx_data:
            # smplx_data = {key: smplx_data[key] for key in smplx_data.files}
            for key in smplx_data.files:
                if isinstance(smplx_data[key], torch.Tensor):
                    smplx_dict[key] = smplx_data[key].cpu()
                else:
                    smplx_dict[key] = smplx_data[key]
        return smplx_dict

    @staticmethod
    def read_gt_keypoints_data(keypoints_path: Path):
        with open(keypoints_path.as_posix(), 'r') as keypoints_file:
            keypoints_data = json.load(keypoints_file)
        return keypoints_data


def nested_dict_update(old_dict, new_dict):
    for key, value in new_dict.items():
        if isinstance(value, collections.abc.Mapping):
            old_dict[key] = nested_dict_update(old_dict.get(key, {}), value)
        else:
            old_dict[key] = value
    return old_dict


def add_dict_to_json(json_filepath: Path, new_data_dict: dict):
    if json_filepath.exists():
        with open(json_filepath, 'r') as json_file:
            data_dict = json.load(json_file)
    else:
        data_dict = {}
    data_dict = nested_dict_update(data_dict, new_data_dict)
    with open(json_filepath, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)
    return


def demo1():
    test_num = 1
    video_name = 'IMG_0014'
    frame_num = 0

    models_dirpath = Path('../data/body_models/smplx')
    smplx_data_path = Path(f'../../runs/testing/test{test_num:04}/{video_name}/frames/{frame_num:04}.npz')
    gt_keypoints_path = Path(f'../../../../research/001_VitPose_Halpe/runs/testing/test0001/{video_name}/json/{frame_num:04}.json')

    mpjppe_computer = MPJPPE(models_dirpath, num_shape_params=10, resolution=(3840, 2160))
    mpjppe = mpjppe_computer.compute_mpjppe(smplx_data_path, gt_keypoints_path)
    print(f'MPJPPE: {mpjppe}')
    return


def demo2():
    """
    On a run folder
    :return:
    """
    test_num = 1
    num_shape_params = 10
    resolution = (3840, 2160)

    models_dirpath = Path('../data/body_models/smplx')
    gt_keypoints_dirpath = Path(f'../../../../research/001_VitPose_Halpe/runs/testing/test0001')
    test_dirpath = Path(f'../../runs/testing/test{test_num:04}')
    qa_computer = MPJPPE(models_dirpath, num_shape_params, resolution)

    qa_data = []
    for video_dirpath in tqdm(sorted(test_dirpath.iterdir())):
        video_name = video_dirpath.stem
        if (not video_dirpath.is_dir()) or video_name in ['quality_scores']:
            continue

        for smplx_data_path in tqdm(sorted(video_dirpath.joinpath('frames').glob('*.npz')), desc=video_name, leave=False):
            frame_num = int(smplx_data_path.stem)
            gt_keypoints_path = gt_keypoints_dirpath / f'{video_name}/json/{frame_num:04}.json'

            qa_score = qa_computer.compute_mpjppe(smplx_data_path, gt_keypoints_path)
            qa_data.append((video_name, frame_num, qa_score))
    qa_data = pandas.DataFrame(qa_data, columns=['video_name', 'frame_num', this_qa_name])
    avg_qa_score = qa_data[this_qa_name].mean().round(4)
    qa_data = qa_data.round({this_qa_name: num_round_off_digits})

    qa_output_path = test_dirpath / f'quality_scores/{this_qa_name}.csv'
    qa_output_path.parent.mkdir(parents=True, exist_ok=True)
    qa_data.to_csv(qa_output_path, index=False)

    qa_avg_path = test_dirpath / f'quality_scores/AverageScores.json'
    qa_avg_data = {this_qa_name: avg_qa_score}
    add_dict_to_json(qa_avg_path, qa_avg_data)
    print(f'Average {this_qa_name}: {avg_qa_score}')
    return


def main():
    demo2()
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
