# Calls demo for images of ModelAgency01 dataset

import argparse
import copy
import datetime
import os
import shutil
import time
import traceback
from pathlib import Path

from demo import call_main

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def execute_shell_command(cmd: str):
    print(cmd)
    os.system(cmd)
    return


def delete_directory(dirpath: Path):
    if dirpath.exists():
        shutil.rmtree(dirpath)
    return


def clean_directory(dirpath: Path):
    delete_directory(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return


def main():
    args = parse_args()

    tmp_dirpath = Path('../../tmp')
    rgb_dirpath = Path(args.rgb_dirpath)
    keypoints_path_format = args.keypoints_path_format
    output_dirpath = Path(args.output_dirpath)

    for agency_dirpath in sorted(rgb_dirpath.iterdir()):
        agency_name = agency_dirpath.stem

        clean_directory(tmp_dirpath)
        tmp_images_dirpath = tmp_dirpath / 'images'
        tmp_keypoints_dirpath = tmp_dirpath / 'keypoints'
        tmp_images_dirpath.mkdir(parents=True, exist_ok=True)
        tmp_keypoints_dirpath.mkdir(parents=True, exist_ok=True)
        for frame_path in sorted(agency_dirpath.iterdir()):
            frame_name = frame_path.stem
            model_index, model_name, frame_num = int(frame_name.split('_')[0]), frame_name.split('_')[1][:-2], int(frame_name.split('_')[1][-2:])
            keypoints_path = Path(keypoints_path_format.format(agency_name=agency_name, frame_name=frame_name))
            tmp_frame_path = tmp_images_dirpath / f'{frame_name}.png'
            tmp_keypoints_path = tmp_keypoints_dirpath / f'{frame_name}.json'
            tmp_frame_path.symlink_to(frame_path)
            tmp_keypoints_path.symlink_to(keypoints_path)

        agency_output_dirpath = output_dirpath / f'{agency_name}'
        agency_output_dirpath.mkdir(parents=True, exist_ok=True)
        demo_args = copy.deepcopy(args)
        del demo_args.rgb_dirpath
        del demo_args.keypoints_path_format
        del demo_args.output_dirpath
        demo_args.output_folder = agency_output_dirpath.as_posix()
        demo_args.exp_opts.append(f'datasets.pose.openpose.data_folder={tmp_dirpath.as_posix()}')
        demo_args.exp_opts.append(f'datasets.pose.openpose.img_folder={tmp_images_dirpath.stem}')
        demo_args.exp_opts.append(f'datasets.pose.openpose.keyp_folder={tmp_keypoints_dirpath.stem}')
        call_main(demo_args)
    return


def parse_args():
    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'PyTorch SMPL-X Regressor Demo'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--exp-cfg', type=str, dest='exp_cfgs',
                        nargs='+',
                        help='The configuration of the experiment')
    parser.add_argument('--rgb-dirpath', dest='rgb_dirpath',
                        default='../../../../../../databases/ModelAgency01/data/test/rgb', type=str,
                        help='The path to the rgb directory')
    parser.add_argument('--keypoints-path-format', dest='keypoints_path_format', type=str,
                        default='../../../../research/001_VitPose_Halpe/runs/testing/test1001/{agency_name}/json/{frame_name}.json',
                        help='The formatted path to directory containing 2D keypoints (agency_name, model_index, model_name and frame_num will be inserted)')
    parser.add_argument('--output-dirpath', dest='output_dirpath',
                        default='../runs/testing/test0000', type=str,
                        help='The folder where the renderings will be saved')
    parser.add_argument('--datasets', nargs='+',
                        default=['openpose'], type=str,
                        help='Datasets to process')
    parser.add_argument('--show', default=False,
                        type=lambda arg: arg.lower() in ['true'],
                        help='Display the results')
    parser.add_argument('--pause', default=-1, type=float,
                        help='How much to pause the display')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
                        nargs='*',
                        help='The configuration of the Detector')
    parser.add_argument('--focal-length', dest='focal_length', type=float,
                        default=5000,
                        help='Focal length')
    parser.add_argument('--save-vis', dest='save_vis', default=False,
                        type=lambda x: x.lower() in ['true'],
                        help='Whether to save visualizations')
    parser.add_argument('--save-mesh', dest='save_mesh', default=False,
                        type=lambda x: x.lower() in ['true'],
                        help='Whether to save meshes')
    parser.add_argument('--save-params', dest='save_params', default=False,
                        type=lambda x: x.lower() in ['true'],
                        help='Whether to save parameters')
    parser.add_argument('--split', default='test', type=str,
                        choices=['train', 'test', 'val'],
                        help='Which split to use')

    args = parser.parse_args()
    return args


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
