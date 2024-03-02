import blenderproc as bproc
import numpy as np
import argparse
import os
import mathutils
import random
from pathlib import Path
import bpy

from blenderproc.python.utility.SetupUtility import SetupUtility

parser = argparse.ArgumentParser()
# parser.add_argument('camera', help="Path to the camera file, should be examples/resources/camera_positions")
parser.add_argument('scene', nargs='?', default="examples/basics/semantic_segmentation/scene.blend", help="Path to the scene.obj file")
parser.add_argument('output_dir', nargs='?', default="examples/basics/semantic_segmentation/output", help="Path to where the final files, will be saved")
parser.add_argument('image_dir', nargs='?', default="images", help="Path to a folder with .jpg textures to be used in the sampling process")
args = parser.parse_args()

bproc.init()

objs = bproc.loader.load_blend(args.scene)

# for j, obj in enumerate(objs):
#     obj.set_cp("category_id", 1)
#     obj.set_cp("instance", "pallet")

# for j, obj in enumerate(objs):
#     if obj.get_cp("category_id") == None:
#         obj.set_cp("category_id", 0)

# light.set_location(bproc.sampler.shell(
#     center=obj.get_location(),
#     radius_min=1,
#     radius_max=5,
#     elevation_min=1,
#     elevation_max=89
# ))
# light.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
# light.set_energy(random.uniform(100, 1000))
# for i in np.linspace(-37.5, 37.5, 10):
#     light = bproc.types.Light()
#     light.set_type("POINT")
#     light.set_location([0, i, 6])
#     light.set_energy(1000)

# poi = bproc.object.compute_poi(bproc.filter.all_with_type(objs, bproc.types.MeshObject))
# objs = bproc.object.convert_to_meshes(objs)
pallets = bproc.filter.all_with_type(bproc.filter.by_cp(objs, "category_id", 1), bproc.types.MeshObject)
# ground = bproc.filter.one_by_attr(objs, "name", "foundation")
# pallets = bproc.object.convert_to_meshes(bproc.filter.by_cp(objs, "category_id", 1))
poi = bproc.object.compute_poi(pallets)

poi_drift = bproc.sampler.random_walk(total_length=50, dims=3, step_magnitude=0.001, window_size=3, interval=[-0.03, 0.03], distribution='uniform')

camera_shaking_rot_angle = bproc.sampler.random_walk(total_length=50, dims=1, step_magnitude=np.pi / 64, window_size=5, interval=[-np.pi / 32, np.pi / 32], distribution='uniform', order=2)
camera_shaking_rot_axis = bproc.sampler.random_walk(total_length=50, dims=3, window_size=10, distribution='normal')
camera_shaking_rot_axis /= np.linalg.norm(camera_shaking_rot_axis, axis=1, keepdims=True)

bproc.camera.set_resolution(1280, 720)

for z_c in np.linspace(0.15, 0.15, 1):
    for i in range(50):
        location_cam = np.array([6 * np.cos(i / 25 * np.pi), 6 * np.sin(i / 25 * np.pi), z_c]) #translation is the camera center from the origin
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi + poi_drift[i] - location_cam)
        R_rand = np.array(mathutils.Matrix.Rotation(camera_shaking_rot_angle[i], 3, camera_shaking_rot_axis[i]))
        rotation_matrix = R_rand @ rotation_matrix
        cam2world_matrix = bproc.math.build_transformation_mat(location_cam, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)

materials = bproc.material.collect_all()

# with open(args.camera, "r") as f:
#     for line in f.readlines():
#         line = [float(x) for x in line.split()]
#         position, euler_rotation = line[:3], line[3:6]
#         matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
#         bproc.camera.add_camera_pose(matrix_world)

# materials = bproc.material.collect_all()
# pallet_materials = bproc.filter.by_cp(materials, "pallet", 1)

# for mat in pallet_materials:
    # mat = mat.nodes[4]
    # mat.set_principled_shader_value("Specular", random.uniform(0, 1))
    # mat.set_principled_shader_value("Roughness", random.uniform(0, 1.0))
    # mat.set_principled_shader_value("Base Color", np.random.uniform([0,0,0,1], [1,1,1,1]))
    # mat.set_principled_shader_value("Metallic", random.uniform(0, 1.0))
    
'''
DONE: reset material of foundation to check whether it renders properly
'''
images = list(Path(args.image_dir).absolute().rglob("Concrete044D_2K-JPG_Roughness.jpg"))
floor_mat = bproc.filter.one_by_cp(materials, "floor", 1)
image = bpy.data.images.load(filepath=str(random.choice(images)))
floor_mat.set_principled_shader_value("Base Color", image)

def sample_pose(obj: bproc.types.MeshObject):
    obj.set_location(np.random.uniform([-2.5, -2.5, 0.1], [2.5, 2.5, 0.1]))
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))
    if obj.get_name() == "Pallet1:pPlane18" or obj.get_name() == "Pallet1:pPlane18.001":
        obj.set_rotation_euler(np.random.uniform([np.pi / 2, 0, 0], [np.pi / 2, 0, np.pi * 2]))
    if obj.get_name() == "pallet":
        obj.set_location(np.random.uniform([-2.5, -2.5, 0], [2.5, 2.5, 0]))

# mesh_objs = bproc.filter.all_with_type(objs, bproc.types.MeshObject)
# mesh_objs = bproc.filter.by_cp(mesh_objs, "category_id", 1)

# bproc.object.sample_poses(
#     pallets,
#     sample_pose_func=sample_pose,
#     objects_to_check_collisions=pallets
# )

# for obj in bproc.object.convert_to_meshes(objs):
#     obj.enable_rigidbody(active=True)
# ground.enable_rigidbody(active=False, collision_shape="MESH")
# bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=4, max_simulation_time=20, check_object_interval=1)

bproc.renderer.set_max_amount_of_samples(50)
# bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"], default_values={"category_id" : 0})
# bproc.renderer.set_output_format(enable_transparency=False)

data = bproc.renderer.render()

bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
                                    instance_segmaps=data["instance_segmaps"],
                                    instance_attribute_maps=data["instance_attribute_maps"],
                                    colors=data["colors"],
                                    color_file_format="JPEG",
                                    append_to_existing_output=False)