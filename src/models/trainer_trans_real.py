import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import drjit as dr
import mitsuba as mi
import numpy as np
import tqdm
from gpytoolbox import remesh_botsch
from omegaconf import DictConfig

from src.models.misc.criterion import *
from src.models.trainer_base import MitsubaTrainer
from src import utils

log = utils.get_pylogger(__name__)

criterion_classes = {'MSE': MSE, 'L1': L1, 'L1Smooth': L1Smooth}

class RealTransientTrainer(MitsubaTrainer):
    def __init__(self,
                 scene_path: str,
                 gt_path: str,
                 shape_to_optimize: List[str],
                 material_to_optimize: List[str],
                 sensor_list: List[List[float]] = None,
                 target_list: List[List[float]] = None,
                 criterion: str = 'MSE',
                 lr_shape: float = 1e-1,
                 lr_material: float = 5e-2,
                 lr_pos: float = 1.0,
                 max_stages: int = 2,  # Now supports 2 stages by default
                 max_iterations: int = 1000,
                 val_interval: int = 100,
                 train_spp: int = 16,
                 val_spp: int = 1024,
                 device: str = 'cuda',
                 lambda_large_steps: float = 25.0):
        """
        Args:
            scene_path: Path to the scene to be trained.
            shape_to_optimize: List of shape names to be optimized.
            material_to_optimize: List of material names to be optimized.
            sensor_list: List of sensor positions.
            target_list: List of target positions.
            criterion: Criterion to be used for training.
            lr_shape: Learning rate for the shape optimizer.
            lr_material: Learning rate for the material optimizer.
            lr_pos: Learning rate for the position optimizer.
            max_stages: Maximum number of stages for training.
            max_iterations: Maximum number of iterations for each stage.
            val_interval: Interval for validation.
            train_spp: Samples per pixel for training.
            val_spp: Samples per pixel for validation.
            device: Device to be used for training.
            lambda_large_steps: Large steps for the optimizer.
        """
        mi.set_variant('cuda_ad_rgb' if device == 'cuda' else 'llvm_ad_rgb')
        import mitransient as mitr
        
        self.scene = mi.load_file(scene_path)
        log.info(f"Loaded scene from {scene_path}.")

        self.scene_path = scene_path
        self.sensor_list= sensor_list
        self.target_list = target_list
        self.up_list = [[0, 1, 0] for _ in range(len(sensor_list))]  # Default up vector
        # Initialize ground truth
        if sensor_list is not None:
            if len(sensor_list) != len(target_list):
                raise ValueError("Sensor list and target list must have the same length!")
            
            self.gt = self.init_multi_view_ground_truth(gt_path, sensor_list, target_list, self.up_list, train_spp)
        else:
            self.gt = self.init_ground_truth(self.scene)

        self.params = mi.traverse(self.scene)
        self.light_initial_pos = self.params['PointLight.position']
        self.shape_to_optimize = shape_to_optimize
        self.material_to_optimize = material_to_optimize

        self.opt_dict = self.init_optimizer(shape_to_optimize, 
                                            material_to_optimize, 
                                            params=self.params,
                                            lambda_large_steps=lambda_large_steps,
                                            lr_shape=lr_shape,
                                            lr_material=lr_material,
                                            lr_pos=lr_pos) # Returns LargeSteps and optimizers
        
        # criterion
        self.criterion = criterion_classes[criterion]()

        # training settings
        self.max_stages = max_stages
        self.max_iterations = max_iterations
        self.val_interval = val_interval
        self.train_spp = train_spp
        self.val_spp = val_spp
        self.manual_optmize = True
        
    def fitting_step(self, idx):
        """
        Perform one optimization step.
        
        Return:
            Dict['loss': float, 'image_vis': np.ndarray]: 
        """
        self.apply_transformation()
        
        if self.sensor_list is not None:
            # Multi-view optimization
            total_loss = mi.Float(0.0)
            image_vis_list = []
            for sensor_idx in range(len(self.sensor_list)):
                # Update sensor position
                origin = self.sensor_list[sensor_idx]
                target = self.target_list[sensor_idx]
                up = self.up_list[sensor_idx]
                transform = mi.Transform4f().look_at(origin=origin, target=target, up=up)
                self.params['sensor.to_world'] = transform
                self.params.update() # Need to update the scene after changing the sensor position

                # Update obj shape
                for obj_name in self.shape_to_optimize:
                    self.params[f'{obj_name}.vertex_positions'] = self.opt_dict['ls_dict'][f'{obj_name}'].from_differential(self.opt_dict['opt_shape'][f'u_{obj_name}'])

                # Updata obj material
                for key in self.material_to_optimize:
                    self.params[key] = self.opt_dict['opt_material'][key]
                # Update light position
                self.params['PointLight.position'] = self.opt_dict['opt_pos']['trans']
                self.params.update()
            
                image, transient = mi.render(self.scene, self.params, spp=self.train_spp)
                # loss = self.criterion(image, self.gt['image_ref'][sensor_idx])
                loss = self.criterion(transient, self.gt['transient_ref'][sensor_idx])
                total_loss += loss
                image_vis_list.append(np.array(mi.util.convert_to_bitmap(image)))
            
            dr.backward(total_loss)
            print("Shape grad:", dr.grad(self.opt_dict['opt_shape']['u_gen_book']))
            print("Material grad:", dr.grad(self.opt_dict['opt_material']['gen_bunny.bsdf.reflectance.value']))
            import pdb; pdb.set_trace()
            self.opt_dict['opt_shape'].step()
            dr.grad_enabled(self.opt_dict['opt_material']['gen_bunny.bsdf.reflectance.value'])
            self.opt_dict['opt_material'].step()
            self.opt_dict['opt_pos'].step()
            loss = total_loss / len(self.sensor_list)
            image_vis = np.vstack(image_vis_list)
            image_vis_gt = np.vstack([np.array(mi.util.convert_to_bitmap(image)) for image in self.gt['image_ref']])
            image_vis = np.hstack([image_vis, image_vis_gt])
        else:
            # Single-view optimization
            #TODO: ALL
            image, transient = mi.render(self.scene, self.params, spp=self.train_spp)
            image = np.array(mi.util.convert_to_bitmap(image))
            image_gt = np.array(mi.util.convert_to_bitmap(self.gt['image_ref']))
            image_vis = np.hstack([image, image_gt])
            loss = self.criterion(transient, self.gt['transient_ref']) 
        
        return {'loss': loss, 'image_vis': image_vis}
    
    def on_stage_start(self, stage_idx):
        """Called at the beginning of each stage"""
        if stage_idx == 1:
            log.info("Starting fine stage - remeshing geometry...")
            self.remesh_geometry()    
    
    def on_stage_end(self, stage_idx):
        """Called at the end of each stage"""
        if stage_idx == 0: 
            log.info("\nCoarse stage complete, preparing for fine stage...")


    @staticmethod # TODO: DONE
    def init_optimizer(shape: List[str], 
                       material: List[str], 
                       params: mi.SceneParameters, 
                       lambda_large_steps: float,
                       lr_shape: float = 1e-1,
                       lr_material: float = 5e-2,
                       lr_pos: float = 1.0) -> Dict[str, Union[mi.ad.Optimizer, Dict[str, mi.ad.LargeSteps]]]:
        """Initialize parameters to be optimized based on keys_to_optimize"""
        ls_dict = {}
        opt_shape = mi.ad.Adam(lr=lr_shape, uniform=True)
        for obj_name in shape:
            ls_dict[f'{obj_name}'] = mi.ad.LargeSteps(params[f'{obj_name}.vertex_positions'],
                                                      params[f'{obj_name}.faces'],
                                                      lambda_large_steps)
            opt_shape[f'u_{obj_name}'] = ls_dict[f'{obj_name}'].to_differential(params[f'{obj_name}.vertex_positions'])

        opt_material = mi.ad.Adam(lr=lr_material, params={key: params[key] for key in material}) # Should include PointLight.intensity.value
        opt_pos = mi.ad.Adam(lr=lr_pos, params={'trans': mi.Point3f(0.0, 0.0, 0.0)})
        return {"opt_shape": opt_shape,
                "opt_material": opt_material,
                "opt_pos": opt_pos,
                "ls_dict": ls_dict}
    
    @staticmethod
    def init_ground_truth(scene) -> Dict[str, Any]:
        """Initialize single-view ground truth"""
        image_ref, transient_ref = mi.render(scene, spp=1024)
        return {'image_ref': image_ref, 'transient_ref': transient_ref}
    
    @staticmethod
    def init_multi_view_ground_truth(gt_path, sensor_list, target_list, up_list, spp) -> Dict[str, Any]:
        """Initialize multi-view ground truth"""
        image_ref_list = []
        transient_ref_list = []
        gt_scene = mi.load_file(gt_path)
        for i in range(len(sensor_list)):
            origin = sensor_list[i]
            target = target_list[i]
            up = up_list[i]
            transform = mi.Transform4f().look_at(origin=origin, target=target, up=up)
            params = mi.traverse(gt_scene)
            params['sensor.to_world'] = transform
            params.update() # Need to update the scene after changing the sensor position

            image_ref, transient_ref = mi.render(gt_scene, spp=spp)
            image_ref_list.append(image_ref)
            transient_ref_list.append(transient_ref)
        return {'image_ref': image_ref_list, 'transient_ref': transient_ref_list}
    
    def apply_transformation(self):
        """Apply light source transformations based on current optimization state"""
        
        opt_pos = self.opt_dict['opt_pos']
        trans = mi.Transform4f().translate([opt_pos['trans'].x,
                                            opt_pos['trans'].y,
                                            opt_pos['trans'].z])
        self.params['PointLight.position'] = dr.ravel(trans @ self.light_initial_pos)        
    
    # TODO: DONE
    def remesh_geometry(self):
        """Remesh the geometry between stages"""
        if not any('vertex_positions' in key for key in self.shape_to_optimize):
            ValueError("No shape components to optimize!")
            return
        if not any('faces' in key for key in self.shape_to_optimize):
            ValueError("No faces available for remeshing!")
            return
        
        for obj_name in self.shape_to_optimize:
            v_np = self.params[f'{obj_name}.vertex_positions'].numpy().reshape((-1,3)).astype(np.float64)
            f_np = self.params[f'{obj_name}.faces'].numpy().reshape((-1,3))

            # Compute average edge length
            l0 = np.linalg.norm(v_np[f_np[:,0]] - v_np[f_np[:,1]], axis=1)
            l1 = np.linalg.norm(v_np[f_np[:,1]] - v_np[f_np[:,2]], axis=1)
            l2 = np.linalg.norm(v_np[f_np[:,2]] - v_np[f_np[:,0]], axis=1)

            target_l = np.mean([l0, l1, l2]) / 2

            v_new, f_new = remesh_botsch(v_np, f_np, i=5, h=target_l, project=True)

            self.params[f'{obj_name}.vertex_positions'] = mi.Float(v_new.flatten().astype(np.float32))
            self.params[f'{obj_name}.faces'] = mi.Int(f_new.flatten())
        
        self.params.update()
        return            
    