# Calls demo for every video iteratively

import copy
import datetime
import time
import traceback

from pathlib import Path
import argparse

from demo import call_main

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def main():
    args = parse_args()
    videos_dirpath = Path(args.videos_dirpath)
    output_dirpath = Path(args.output_dirpath)
    for video_frames_dirpath in sorted(videos_dirpath.iterdir()):
        video_name = video_frames_dirpath.stem
        video_output_dirpath = output_dirpath / f'{video_name}/frames'
        video_output_dirpath.mkdir(parents=True, exist_ok=True)
        demo_args = copy.deepcopy(args)
        del demo_args.videos_dirpath
        del demo_args.output_dirpath
        demo_args.output_folder = video_output_dirpath.as_posix()
        demo_args.exp_cfg.append(f'datasets.pose.openpose.data_folder={video_frames_dirpath.as_posix()}')
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
    parser.add_argument('--videos-dirpath', dest='videos_dirpath',
                        default='../../../../../../databases/spree_internal/data/rgb_png', type=str,
                        help='The folder where the renderings will be saved')
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
