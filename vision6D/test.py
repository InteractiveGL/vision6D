import json
import pathlib
import numpy as np
import vision6D as vis

if __name__ == "__main__":
    root = pathlib.Path.cwd()
    with open(root / "vision6D" / "ossiclesCoordinateMapping.json", "r") as f: data = json.load(f)
    verts = np.array(data['verts']).reshape((len(data['verts'])), 3)
    faces = np.array(data['faces']).reshape((len(data['faces'])), 3)

    longitude = np.array(data['longitude']).reshape((len(data['longitude'])), 1)
    latitude = np.array(data['latitude']).reshape((len(data['latitude'])), 1)

    # read atlas mesh
    mesh = vis.utils.load_trimesh(vis.config.OSSICLES_MESH_PATH_5997_right)
    print("hhh")