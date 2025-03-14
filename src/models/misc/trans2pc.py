import numpy as np
import mitsuba as mi 
from typing import Union
import os 
mi.set_variant('cuda_ad_rgb')
import mitransient as mitr
from tqdm import tqdm

def multi_view_generate_point_cloud_synthetic(scene_path : str,
                                              origin_list: np.array, 
                                              target_list: np.array,
                                              up: np.array, 
                                              fov: Union[int, float], 
                                              resolution: tuple,
                                              spp: int = 512) -> None:
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

    for i in tqdm(range(num_views), desc="Generating point cloud"):

        origin = origin_list[i]
        target = target_list[i]

        transform = mi.Transform4f().look_at(origin=origin, target=target, up=up)
        params['sensor.to_world'] = transform
        params['PointLight.position'] = origin
        params.update()
        _, data_transient = mi.render(scene, spp=spp)
        data_transient_np = np.sum(np.array(data_transient), axis=-1)
        
        depth_map = np.argmax(data_transient_np, axis=2) * bin_width_opl
            
        forward = target - origin
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

        point_list.append(point_cloud)

    point_list = np.concatenate(point_list, axis=0)

if __name__ == '__main__':
    # 加载场景
    scene = mi.load_file(os.path.abspath('cornell-box/cbox_diffuse.xml'))
    params = mi.traverse(scene)

    # 定义参数
    fov = 50
    resolution = (400, 400)
    up = np.array([0, 1, 0])

    # 相机 1
    origin_1 = np.array([278, 278, -800])  
    target_1 = np.array([278, 278, -799])  
    transform_1 = mi.Transform4f().look_at(origin=origin_1, target=target_1, up=up)
    params['sensor.to_world'] = transform_1
    params['PointLight.position'] = origin_1
    params.update()
    data_steady, data_transient = mi.render(scene, spp=512)
    data_transient_np = np.sum(np.array(data_transient), axis=-1)
    depth_1 = np.argmax(data_transient_np, axis=2)
    depth_1 = depth_1 * 4
    points_1 = generate_point_cloud(origin_1, target_1, up, fov, resolution, depth_1)

    # 相机 2
    origin_2 = np.array([-800, 278, 278])  
    target_2 = np.array([-799, 278, 278])  
    transform_2 = mi.Transform4f().look_at(origin=origin_2, target=target_2, up=up)
    params['sensor.to_world'] = transform_2
    params['PointLight.position'] = origin_2
    params.update()
    data_steady, data_transient = mi.render(scene, spp=512)
    data_transient_np = np.sum(np.array(data_transient), axis=-1)
    depth_2 = np.argmax(data_transient_np, axis=2)
    depth_2 = depth_2 * 4
    points_2 = generate_point_cloud(origin_2, target_2, up, fov, resolution, depth_2)

    # 相机 3
    origin_3 = np.array([278, 278, 1200])  
    target_3 = np.array([278, 278, 1199])  
    transform_3 = mi.Transform4f().look_at(origin=origin_3, target=target_3, up=up)
    params['sensor.to_world'] = transform_3
    params['PointLight.position'] = origin_3
    params.update()
    data_steady, data_transient = mi.render(scene, spp=512)
    data_transient_np = np.sum(np.array(data_transient), axis=-1)
    depth_3 = np.argmax(data_transient_np, axis=2)
    depth_3 = depth_3 * 4
    points_3 = generate_point_cloud(origin_3, target_3, up, fov, resolution, depth_3)

    # 相机 4
    origin_4 = np.array([1200, 278, 278])  
    target_4 = np.array([1199, 278, 278])  
    transform_4 = mi.Transform4f().look_at(origin=origin_4, target=target_4, up=up)
    params['sensor.to_world'] = transform_4
    params['PointLight.position'] = origin_4
    params.update()
    data_steady, data_transient = mi.render(scene, spp=512)
    data_transient_np = np.sum(np.array(data_transient), axis=-1)
    depth_4 = np.argmax(data_transient_np, axis=2)
    depth_4 = depth_4 * 4
    points_4 = generate_point_cloud(origin_4, target_4, up, fov, resolution, depth_4)

    # 合并点云
    import open3d as o3d
    point_cloud = o3d.geometry.PointCloud()
    points = np.concatenate((points_1, points_2, points_3, points_4), axis=0)
    point_cloud.points = o3d.utility.Vector3dVector(points)
    downsampled_cloud = point_cloud.random_down_sample(sampling_ratio=0.004)
    # pc = down_10x, pc_down_10x = down_100x
    # 保存点云
    o3d.io.write_point_cloud("pc.ply", downsampled_cloud)