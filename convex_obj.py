import os
import tempfile
import trimesh
import pybullet as p
import argparse
import json
import shutil


def vhacd_decompose(mesh_file, output_dir=None, suffix='.obj'):
    if output_dir is None:
        output_dir = os.path.dirname(mesh_file)
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f"Processing {mesh_file} with vhacd ...")
    obj_filename = os.path.join(output_dir, ''.join(os.path.basename(mesh_file).split('.')[:-1] + [suffix]))
    mesh = trimesh.load(mesh_file, force="mesh")
    mesh.export(obj_filename)
    with tempfile.NamedTemporaryFile(delete=True) as log_file:
        kwargs = {}
        p.vhacd(obj_filename, obj_filename, log_file.name, **kwargs)
    print(f"Decomposed mesh saved to {obj_filename}")
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    output_root_folder = "./asset/objects"
    obj_categ_list = os.listdir(output_root_folder)
    for obj_categ in obj_categ_list:
        obj_categ_path = os.path.join(output_root_folder, obj_categ)
        obj_list = os.listdir(obj_categ_path)
        for obj_name in obj_list:
            if obj_name.endswith(".glb"):
                output_obj_name = obj_name.replace(".glb", ".obj")
                input_file = os.path.join(obj_categ_path, obj_name)
                output_file = os.path.join(obj_categ_path, output_obj_name)
                print(f"Processing file: {input_file}")
                if not os.path.exists(output_file):
                    vhacd_decompose(input_file, obj_categ_path)
                else:
                    print(f"Already processed: {output_file}, skip.")



