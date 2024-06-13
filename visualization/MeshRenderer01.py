# Renders the mesh for the SMPL-X model paprameters output by the model
# Uses the mesh renderer from Phong's code

import colorsys
import datetime
import os
import time
import traceback
from pathlib import Path

import cv2
import numpy
import numpy as np
import pyrender
import skimage.io
import smplx
import torch
import trimesh
from scipy.spatial.transform import Rotation
from smplx import SMPLXLayer

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class Renderer(object):
    def __init__(self, focal_length=600, img_w=512, img_h=512, camera_center=None, faces=None, same_mesh_color=False):
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w, viewport_height=img_h, point_size=1.0)
        if camera_center is not None:
            self.camera_center = camera_center
        else:
            self.camera_center = [img_w // 2, img_h // 2]
        self.focal_length = focal_length
        self.faces = faces
        self.same_mesh_color = same_mesh_color
        return

    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(0, 0, 0, 0)):
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 0)
        # Create camera. Camera will always be at [0,0,0]
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                  cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=np.eye(4))

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        # for DirectionalLight, only rotation matters
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        scene.add(light, pose=light_pose)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        scene.add(light, pose=light_pose)

        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # multiple person
        num_people = len(verts)
        # for every person in the scene
        for n in range(num_people):
            mesh = trimesh.Trimesh(verts[n], self.faces)
            mesh.apply_transform(rot)
            if self.same_mesh_color:
                mesh_color = colorsys.hsv_to_rgb(0.6, 0.5, 1.0)
            else:
                mesh_color = colorsys.hsv_to_rgb(float(n) / num_people, 0.5, 1.0)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False)
            scene.add(mesh, 'mesh')

        # Alpha channel was not working previously, need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_rgb = color_rgba[:, :, :3]
        if bg_img_rgb is None:
            return color_rgb
        else:
            mask = depth_map > 0
            bg_img_rgb[mask] = color_rgb[mask]
            return bg_img_rgb

    def render_side_view(self, verts):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid
        side_view = self.render_front_view(pred_vert_arr_side)
        return side_view

    def delete(self):
        """
        Need to delete before creating the renderer next time
        """
        self.renderer.delete()


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

    shape_params_tr = torch.from_numpy(shape_params).float().unsqueeze(0)
    pose_matrices_tr = torch.from_numpy(pose_matrices).float().unsqueeze(0)

    # pose_axis_angles = get_rotations_as_axis_angles(pose_matrices)
    # pose_axis_angles_tr = torch.from_numpy(pose_axis_angles).float().unsqueeze(0)
    # smplx_model = smplx.create(models_dirpath.parent.as_posix(), model_type='smplx', num_betas=num_shape_params)
    # smplx_output = smplx_model(betas=shape_params_tr, body_pose=pose_axis_angles_tr[:, 1:], global_orient=pose_axis_angles_tr[:, :1], return_verts=True)

    smplx_model = SMPLXLayer(models_dirpath.as_posix(), num_betas=num_shape_params)
    smplx_output = smplx_model(betas=shape_params_tr, body_pose=pose_matrices_tr[:, 1:],
                               global_orient=pose_matrices_tr[:, :1], pose2rot=False)

    faces = smplx_model.faces
    raw_vertices = smplx_output.vertices
    translated_vertices = raw_vertices + camera_translation

    renderer = Renderer(focal_length=focal_length, img_w=w, img_h=h, camera_center=camera_center, faces=faces,
                        same_mesh_color=False)
    front_view = renderer.render_front_view(translated_vertices.cpu().numpy(), bg_img_rgb=image.copy())
    save_image(Path('./front_view.png'), front_view)
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
