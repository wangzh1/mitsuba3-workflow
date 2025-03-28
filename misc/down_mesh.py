import trimesh
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", "-i", type=str, required=True)
parser.add_argument("--output_path", "-o", type=str, required=True)
parser.add_argument("--faces", "-f", type=int, required=True)
args = parser.parse_args()

def downsample_obj_by_ratio(input_path, output_path, target_faces):
    """
    faces: int
        The target # of faces.
    """

    mesh = trimesh.load(input_path)
    
    original_faces = len(mesh.faces)
    print(f"Original faces: {original_faces}")
    print(f'Target faces: {target_faces}')
    
    ratio = 1 - target_faces / original_faces

    simplified_mesh = mesh.simplify_quadric_decimation(ratio)
    print(f"Downsampled faces: {len(simplified_mesh.faces)}")
    simplified_mesh.export(output_path)

if __name__ == "__main__":
    downsample_obj_by_ratio(args.input_path, args.output_path, args.faces)