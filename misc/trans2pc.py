import os
import cv2
from typing import Union

import mitsuba as mi
import scipy.io as sio
import numpy as np
import open3d as o3d

mi.set_variant('cuda_ad_rgb')
import mitransient as mitr
from tqdm import tqdm


def multi_view_generate_point_cloud_synthetic(scene_path : str,
                                              origin_list: np.array, 
                                              target_list: np.array,
                                              up: np.array, 
                                              fov: Union[int, float], 
                                              resolution: tuple,
                                              spp: int = 512,
                                              down_rate: float = 1.0,
                                              point_cloud_path: str = None) -> None:
    """
    Generate point cloud from depth map.
    
    params:
        scene_path (str): Path to the Mitsuba scene file, CAMERA_UNWARP=TRUE.
        origin_list (np.array): Camera position of all viewpoints.
        target_list (np.array): Camera target of all viewpoints.
        up (np.array): Camera up direction, temporally set to be fixed.
        fov (float): Field of view in degrees, temporally set to be fixed.
        resolution (tuple): Image resolution (width, height), temporally set to be fixed.
    return:
        None, save the point cloud to a file.
    """
    scene = mi.load_file(os.path.abspath(scene_path))
    params = mi.traverse(scene)
    bin_width_opl = params['sensor.film.bin_width_opl']

    if not params:
        raise ValueError("Failed to traverse the scene parameters.")
    if 'sensor.to_world' not in params:
        raise ValueError("Sensor not found in the scene.")
    if 'PointLight.position' not in params:
        raise ValueError("Point light source not found in the scene.")
    
    width, height = resolution
    fov_rad = np.deg2rad(fov)  
    num_views = len(origin_list)
    point_list = []
    color_list = []

    for i in tqdm(range(num_views), desc="Generating point cloud"):

        origin = origin_list[i]
        target = target_list[i]

        transform = mi.Transform4f().look_at(origin=origin, target=target, up=up)
        params['sensor.to_world'] = transform
        params['PointLight.position'] = origin
        params.update()
        data_steady, data_transient = mi.render(scene, spp=spp)
        if i == 0:
            bitmap = mi.util.convert_to_bitmap(data_steady)
            save_img_name = point_cloud_path.split('.')[0] + '.png'
            cv2.imwrite(save_img_name, cv2.cvtColor(np.array(bitmap), cv2.COLOR_RGB2BGR))
        
        depth_map = np.argmax(np.sum(np.array(data_transient), axis=-1), axis=2) * bin_width_opl
        
        target, origin, up = np.array(target), np.array(origin), np.array(up)
        forward = np.array(target) - np.array(origin)
        forward = forward / np.linalg.norm(forward) 
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)  
        up = np.cross(right, forward) 

        aspect_ratio = width / height
        plane_height = 2 * np.tan(fov_rad / 2)  
        plane_width = plane_height * aspect_ratio 

        x = np.linspace(-plane_width / 2, plane_width / 2, width)
        y = np.linspace(plane_height / 2, -plane_height / 2, height)
        xx, yy = np.meshgrid(x, y)

        directions = (
            forward[np.newaxis, np.newaxis, :] +
            right[np.newaxis, np.newaxis, :] * xx[:, :, np.newaxis] +
            up[np.newaxis, np.newaxis, :] * yy[:, :, np.newaxis]
        )
        directions = directions / np.linalg.norm(directions, axis=2, keepdims=True) 

        point_cloud = origin + directions * depth_map[:, :, np.newaxis] 

        point_cloud = point_cloud.reshape(-1, 3)
        valid_mask = np.linalg.norm(point_cloud - origin[np.newaxis, :], axis=1) > 1e-3
        point_cloud = point_cloud[valid_mask]
        
        point_list.append(point_cloud)

    point_list = np.concatenate(point_list, axis=0)
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_list)

    if down_rate < 1.0:
        point_cloud = point_cloud.random_down_sample(sampling_ratio=down_rate)
    o3d.io.write_point_cloud(point_cloud_path, point_cloud)

def multi_view_generate_point_cloud_real(transient_dir: str,
                                         fov: Union[int, float],
                                         resolution: tuple,
                                         bin_width_opl: float = 0.0012, # 4ps = 0.0012m
                                         down_rate: float = 1.0,
                                         point_cloud_path: str = None) -> None:
    """"
    Generate point cloud from transient data.
    params:
        transient_dir (str): Path to the Mitsuba transient data.
        origin_list (np.array): Camera position of all viewpoints.
        target_list (np.array): Camera target of all viewpoints.
        up (np.array): Camera up direction, temporally set to be fixed.
        fov (float): Field of view in degrees, temporally set to be fixed.
        resolution (tuple): Image resolution (width, height), temporally set to be fixed.
        down_rate (float): Downsample rate for the point cloud.
    return:
        None, save the point cloud to a file.
    """
    if not os.path.exists(transient_dir):
        raise ValueError(f"Transient directory {transient_dir} does not exist.")
    if not os.path.isdir(transient_dir):
        raise ValueError(f"Transient directory {transient_dir} is not a directory.")
    if not os.listdir(transient_dir):
        raise ValueError(f"Transient directory {transient_dir} is empty.")

    
    width, height = resolution
    fov = 24
    fov_rad = np.deg2rad(fov)  
    transient_list = sorted(os.listdir(transient_dir))
    num_views = len(transient_list)
    point_list = []

    for i in tqdm(range(num_views), desc="Generating point cloud"):
        # if i >= 1:
            # continue
        # if not (transient_list[i].endswith('_0.mat') or transient_list[i].endswith('_180.mat')):
        #     continue
        data = sio.loadmat(os.path.join(transient_dir, transient_list[i]))
        origin = data['camera_coordinate'][:, 0]
        # import pdb; pdb.set_trace()
        # origin *= 1.01
        target = data['target_vector'][:, 0]
        up = data['up_vector'][:, 0]
        data_transient = np.roll(data['shifted_data'], shift=-30, axis=1).reshape(width, height, -1)
        # data_transient = data['shifted_data'].reshape(width, height, -1)
        

        # data_transient = load(os.path.join(transient_dir, f"{i:02d}"))

        # import pdb; pdb.set_trace()
        depth_map = np.argmax(data_transient, axis=2) / 2 * bin_width_opl
        depth_map = depth_map.T # MATLAB convert to numpy
        depth_map = np.flip(depth_map, axis=1)
        # depth_map[...] = -origin[2] * 0.91
        forward = target
        forward = target / np.linalg.norm(target) 
        # import pdb; pdb.set_trace()
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)  
        up = np.cross(right, forward) 

        aspect_ratio = width / height
        plane_height = 2 * np.tan(fov_rad / 2)
        plane_width = plane_height * aspect_ratio 

        x = np.linspace(-plane_width / 2, plane_width / 2, width)
        y = np.linspace(plane_height / 2, -plane_height / 2, height)
        xx, yy = np.meshgrid(x, y)

        directions = (
            forward[np.newaxis, np.newaxis, :] +
            right[np.newaxis, np.newaxis, :] * xx[:, :, np.newaxis] +
            up[np.newaxis, np.newaxis, :] * yy[:, :, np.newaxis]
        )
        # import pdb; pdb.set_trace()
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True) 
        # directions[...] = np.array([0, 0, 1])
        point_cloud = origin + directions * depth_map[:, :, np.newaxis]

        point_cloud = point_cloud.reshape(-1, 3)
        # distance in meters
        valid_mask = np.linalg.norm(point_cloud - origin[np.newaxis, :], axis=1) < 100e-2
        point_cloud = point_cloud[valid_mask]
        valid_mask = np.linalg.norm(point_cloud - origin[np.newaxis, :], axis=1) > 50e-2
        point_cloud = point_cloud[valid_mask]
        point_list.append(point_cloud)

    point_list = np.concatenate(point_list, axis=0)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_list)
    if down_rate < 1.0:
        point_cloud = point_cloud.random_down_sample(sampling_ratio=down_rate)
    o3d.io.write_point_cloud(point_cloud_path, point_cloud)
if __name__ == '__main__':
    scene_path = os.path.abspath('transient_scenes/synthetic/cornell-box/cbox_diffuse.xml')
    # # scene = mi.load_file(os.path.abspath('cornell-box/cbox_diffuse.xml'))
    multi_view_generate_point_cloud_synthetic(scene_path=scene_path,
                                            origin_list=[[278, 278, -800], [-800, 278, 278], [278, 278, 1200], [1200, 278, 278]],
                                            target_list=[[278, 278, -799], [-799, 278, 278], [278, 278, 1199], [1199, 278, 278]],
                                            up=[0, 1, 0],
                                            fov=40,
                                            resolution=(400, 400),
                                            spp=512,
                                            down_rate=0.5,
                                            point_cloud_path="book.ply")
    # multi_view_generate_point_cloud_real(transient_dir='/Users/wangzh/Desktop/ShanghaiTech/Spring_2025/ICCP2025/mitsuba-tutorials/mitransient-1.1.0/examples/transient/frog',
    #                                      fov=24,
    #                                      resolution=(128, 128),
    #                                      bin_width_opl=0.0012 * 8, # 4ps = 0.0012m
    #                                      down_rate=1.0,
    #                                      point_cloud_path='frog.ply')
