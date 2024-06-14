# Adds quality scores on the image obtained by rendering the mesh on top of the image

import datetime
import time
import traceback

from pathlib import Path

import cv2
import pandas
import skimage.io
from tqdm import tqdm

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def read_image(image_path: Path):
    image = skimage.io.imread(image_path)
    return image


def save_image(image_path: Path, image):
    skimage.io.imsave(image_path, image)
    return


def main():
    test_num = 1
    qa_name = 'MPJPPE02'

    test_dirpath = Path(f'../../runs/testing/test{test_num:04}')
    qa_scores_path = next(test_dirpath.glob(f'quality_scores/{qa_name}*.csv'))
    qa_scores_data = pandas.read_csv(qa_scores_path)

    for video_dirpath in tqdm(sorted(test_dirpath.iterdir())):
        video_name = video_dirpath.stem
        if (not video_dirpath.is_dir()) or video_name in ['quality_scores']:
            continue

        for image_mesh_path in tqdm(sorted(video_dirpath.glob('frames/*_hd_stage_02_overlay.png')), desc=video_name, leave=False):
            frame_num = int(image_mesh_path.stem[:4])
            output_path = video_dirpath / f'frames/{frame_num:04}_image_mesh_quality.png'
            if output_path.exists():
                continue

            qa_score = qa_scores_data[(qa_scores_data['video_name'] == video_name) & (qa_scores_data['frame_num'] == frame_num)][qa_name].values[0]
            image_mesh = read_image(image_mesh_path)
            cv2.putText(image_mesh, qa_name, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 128, 128, 255), 3, cv2.LINE_AA)
            cv2.putText(image_mesh, str(qa_score), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 128, 128, 255), 3, cv2.LINE_AA)
            save_image(output_path, image_mesh)
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
