import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt # We'll also want to plot some outputs
import os

mi.set_variant('cuda_ad_rgb')
from mitsuba import ScalarTransform4f as T
import numpy as np
sensor_count = 8
sensors = []

golden_ratio = (1 + 5**0.5)/2

origin_list=[[278, 200, -800], [-800, 200, 278], [278, 200, 1200], [1200, 200, 278]]
target_list=[[278, 200, -799], [-799, 200, 278], [278, 200, 1199], [1199, 200, 278]]

for i in range(4):
    # theta = 2 * dr.pi * i / golden_ratio
    # phi = dr.acos(1 - 2*(i+0.5)/sensor_count)
    
    # d = 300
    origin = origin_list[i]
    target = target_list[i]
    up = [0, 1, 0]
    
    sensors.append(mi.load_dict({
        'type': 'perspective',
        'fov': 30,
        'to_world': T().look_at(target=target, origin=origin, up=up),
        'film': {
            'type': 'hdrfilm',
            'width': 256, 'height': 256,
            'filter': {'type': 'gaussian'},
            'sample_border': True,
        },     
        'sampler': {
            'type': 'independent',
            'sample_count': 128
        },
    }))

    scene_dict = {
    'type': 'scene',
    'integrator': {
        'type': 'direct_projective',
        # Indirect visibility effects aren't that important here
        # let's turn them off and save some computation time
        'sppi': 0, 
    },
    # 'emitter': {
    #     'type': 'envmap',
    #     'filename': "transient_scenes/synthetic/textures/envmap2.exr",
    # },
    'shape': {
        'type': 'obj',
        'filename': "transient_scenes/synthetic/cornell-box/meshes/apple.obj",
        'bsdf': {'type': 'diffuse'}
    }
}

scene_target = mi.load_dict(scene_dict)
ref_images = [mi.render(scene_target, sensor=sensors[i], spp=256) for i in range(4)]
scene_dict['shape']['filename'] = 'transient_scenes/synthetic/cornell-box/meshes/gen_apple.obj'
scene_source = mi.load_dict(scene_dict)
init_imgs = [mi.render(scene_source, sensor=sensors[i], spp=128) for i in range(4)]
params = mi.traverse(scene_source)
lambda_ = 25
ls = mi.ad.LargeSteps(params['shape.vertex_positions'], params['shape.faces'], lambda_)

opt = mi.ad.Adam(lr=1e-1, uniform=True)
opt['u'] = ls.to_differential(params['shape.vertex_positions'])
iterations = 100
for it in range(iterations):
    total_loss = mi.Float(0.0)
    
    for sensor_idx in range(4):
        # Retrieve the vertex positions from the latent variable
        params['shape.vertex_positions'] = ls.from_differential(opt['u'])
        params.update()
        
        img = mi.render(scene_source, params, sensor=sensors[sensor_idx], seed=it)
        
        # L1 Loss
        loss = dr.mean(dr.abs(img - ref_images[sensor_idx]))
        
        dr.backward(loss)
        opt.step()
        
        total_loss += loss
        
    print(f"Iteration {1+it:03d}: Loss = {total_loss}", end='\r')

params.update()

for i in range(4):
    image = mi.render(scene_source, sensor=sensors[i], spp=256)
    mi.util.write_bitmap(f"bunny_coarse_{i}.png", image)

import numpy as np
v_np = params['shape.vertex_positions'].numpy().reshape((-1,3)).astype(np.float64)
f_np = params['shape.faces'].numpy().reshape((-1,3))

# Compute average edge length
l0 = np.linalg.norm(v_np[f_np[:,0]] - v_np[f_np[:,1]], axis=1)
l1 = np.linalg.norm(v_np[f_np[:,1]] - v_np[f_np[:,2]], axis=1)
l2 = np.linalg.norm(v_np[f_np[:,2]] - v_np[f_np[:,0]], axis=1)

target_l = np.mean([l0, l1, l2]) / 2


from gpytoolbox import remesh_botsch

v_new, f_new = remesh_botsch(v_np, f_np, i=5, h=target_l, project=True)

params['shape.vertex_positions'] = mi.Float(v_new.flatten().astype(np.float32))
params['shape.faces'] = mi.Int(f_new.flatten())
params.update()

ls = mi.ad.LargeSteps(params['shape.vertex_positions'], params['shape.faces'], lambda_)
opt = mi.ad.Adam(lr=1e-1, uniform=True)
opt['u'] = ls.to_differential(params['shape.vertex_positions'])

integrator = mi.load_dict({
    'type': 'direct_projective'
})

iterations = 100 
for it in range(iterations):
    total_loss = mi.Float(0.0)
    for sensor_idx in range(4):
        # Retrieve the vertex positions from the latent variable
        params['shape.vertex_positions'] = ls.from_differential(opt['u'])
        params.update()

        img = mi.render(scene_source, params, sensor=sensors[sensor_idx], seed=it, integrator=integrator)
    
        # L1 Loss
        loss = dr.mean(dr.abs(img - ref_images[sensor_idx]))
        dr.backward(loss)
        total_loss += loss
    opt.step()

    print(f"Iteration {1+it:03d}: Loss = {total_loss}", end='\r')

params['shape.vertex_positions'] = ls.from_differential(opt['u'])
params.update()

for i in range(4):
    image = mi.render(scene_source, sensor=sensors[i], spp=256)
    mi.util.write_bitmap(f"bunny_fine_{i}.png", image)

v_np = params['shape.vertex_positions'].numpy().reshape((-1, 3))
f_np = params['shape.faces'].numpy().reshape((-1, 3))

import trimesh
mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
mesh.export("optimized_mesh.obj")  # 保存为 OBJ 文件