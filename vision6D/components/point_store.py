import pathlib

import numpy as np
import pyvista as pv
import trimesh

from . import Singleton
from ..tools import utils

class PointStore(metaclass=Singleton):
    def __init__(self):
        self.point_path = None
        self.point_data = None
        self.point_name = None
        self.point_actors = {}

    def reset(self):
        self.point_path = None
        self.point_data = None
        self.point_name = None
        self.point_actors.clear()

    def remove_point(self, name):
        del self.point_actors[name]

    def load_points(self, point_source):
        if isinstance(point_source, pathlib.Path) or isinstance(point_source, str):
            self.point_path = str(point_source)
            self.point_name = pathlib.Path(self.point_path).stem + "_point"
            if pathlib.Path(point_source).suffix == '.npy': point_source = np.load(point_source)
            elif pathlib.Path(point_source).suffix == '.mesh': point_source = utils.load_trimesh(point_source)
            else: point_source = pv.read(point_source)

        if isinstance(point_source, np.ndarray):
            if point_source.shape[-1] == 2: 
                self.point_data = point_source
                
        if isinstance(point_source, trimesh.Trimesh):
            point_source.vertices = point_source.vertices.reshape(-1, 3)
            self.point_data = point_source.vertices
            
        if isinstance(point_source, pv.PolyData):
            self.point_data = point_source.points
            

        