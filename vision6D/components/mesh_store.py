'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: mesh_store.py
@time: 2023-07-03 20:24
@desc: create store for mesh related base functions
'''

import pathlib
from typing import Optional, Dict, List
from dataclasses import dataclass, field

import trimesh
import pyvista as pv
import numpy as np

from . import Singleton
from ..tools import utils

@dataclass
class MeshData:
    mesh_path: str
    name: str
    source_mesh: trimesh.Trimesh
    pv_mesh: pv.PolyData
    actor: pv.Actor
    opacity_spinbox: Optional[str]
    color_button: Optional[str]
    color: str
    texture_path: Optional[str] = None
    spacing: List[float] = field(default_factory=[1, 1, 1])
    mirror_x: bool = False
    mirror_y: bool = False
    opacity: float = 0.9
    previous_opacity: float = 0.9
    initial_pose: np.ndarray = np.eye(4)
    undo_poses: List[np.ndarray] = field(default_factory=list)
    undo_vertices: List[np.ndarray] = field(default_factory=list)

class MeshStore(metaclass=Singleton):
    def __init__(self):
        self.reference: Optional[str] = None
        self.meshes: Dict[str, MeshData] = {}
        self.color_counter = 0
        self.color_button = None
        self.colors = ["wheat", "cyan", "magenta", "yellow", "lime", "dodgerblue", "white", "black"]
        self.latlon = utils.load_latitude_longitude()

    def reset(self): 
        self.mesh_path = None
        self.color_counter = 0
        self.meshes.clear()

    #^ Mesh related
    def add_mesh(self, mesh_source) -> Optional[MeshData]:

        source_mesh = None

        if isinstance(mesh_source, pathlib.Path) or isinstance(mesh_source, str):
            mesh_path = str(mesh_source)
            if pathlib.Path(mesh_path).suffix == '.mesh': mesh_source = utils.load_trimesh(mesh_source)
            else: mesh_source = pv.read(mesh_path)

        if isinstance(mesh_source, trimesh.Trimesh):
            source_mesh = mesh_source
            source_mesh.vertices = source_mesh.vertices.reshape(-1, 3)
            source_mesh.faces = source_mesh.faces.reshape(-1, 3)
            pv_mesh = pv.wrap(mesh_source)

        if isinstance(mesh_source, pv.PolyData):
            source_mesh = trimesh.Trimesh(mesh_source.points, mesh_source.faces.reshape((-1, 4))[:, 1:], process=False)
            source_mesh.vertices = source_mesh.vertices.reshape(-1, 3)
            source_mesh.faces = source_mesh.faces.reshape(-1, 3)
            pv_mesh = pv.wrap(source_mesh)
        
        if source_mesh is not None:
            name = pathlib.Path(mesh_path).stem
            while name in self.meshes.keys(): name += "_copy"
            mesh_data = MeshData(mesh_path=mesh_path,
                                name=name, 
                                source_mesh=source_mesh, 
                                pv_mesh=pv_mesh,
                                color_button=None,
                                actor=None,
                                color=self.colors[self.color_counter],
                                texture_path=None,
                                opacity_spinbox=None,
                                spacing=[1, 1, 1])
            
            # set spacing for the mesh
            mesh_data.pv_mesh.points *= mesh_data.spacing

            self.meshes[mesh_data.name] = mesh_data

            # assign a color to every mesh
            self.color_counter += 1
            self.color_counter %= len(self.colors)

            return mesh_data

        return None

    def remove_mesh(self, name):
        del self.meshes[name]
        self.reference = None
    
    def get_poses_from_undo(self, mesh_data):
        transformation_matrix = mesh_data.undo_poses.pop()
        while mesh_data.undo_poses and (transformation_matrix == mesh_data.actor.user_matrix).all(): 
            transformation_matrix = mesh_data.undo_poses.pop()
        mesh_data.actor.user_matrix = transformation_matrix
            
    def undo_actor_pose(self, name):
        mesh_data = self.meshes[name]
        self.get_poses_from_undo(mesh_data)
            
