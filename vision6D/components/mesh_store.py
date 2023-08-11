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
    name: str
    path: str
    source_mesh: trimesh.Trimesh
    pv_mesh: pv.PolyData
    actor: pv.Actor
    color: str
    spacing: List[float] = field(default_factory=list)
    pose_path: str = ''
    mirror_x: bool = False
    mirror_y: bool = False
    opacity: float = 0.3
    previous_opacity: float = 0.3
    transformation_matrix: np.ndarray = np.eye(4)
    initial_pose: np.ndarray = np.eye(4)
    undo_poses: List[np.ndarray] = field(default_factory=list)

class MeshStore(metaclass=Singleton):
    def __init__(self, window_size):
        self.reference: Optional[str] = None
        self.render = utils.create_render(window_size[0], window_size[1])
        self.meshes: Dict[str, MeshData] = {}
        self.color_counter = 0
        self.colors = ["cyan", "magenta", "yellow", "lime", "dodgerblue", "darkviolet", "darkorange", "darkgrey"]
        self.latlon = utils.load_latitude_longitude()

    def reset(self): 
        self.color_counter = 0
        self.meshes.clear()

    #^ Mesh related
    def add_mesh(self, mesh_source) -> Optional[MeshData]:

        source_mesh = None

        if isinstance(mesh_source, pathlib.WindowsPath) or isinstance(mesh_source, str):
            mesh_path = str(mesh_source)
            if pathlib.Path(mesh_path).suffix == '.mesh': mesh_source = utils.load_trimesh(mesh_source)
            else: mesh_source = pv.read(mesh_path)

        if isinstance(mesh_source, trimesh.Trimesh):
            source_mesh = mesh_source
            assert (source_mesh.vertices.shape[1] == 3 and source_mesh.faces.shape[1] == 3), "it should be N by 3 matrix"
            pv_mesh = pv.wrap(mesh_source)

        if isinstance(mesh_source, pv.PolyData):
            source_mesh = trimesh.Trimesh(mesh_source.points, mesh_source.faces.reshape((-1, 4))[:, 1:], process=False)
            assert (source_mesh.vertices.shape[1] == 3 and source_mesh.faces.shape[1] == 3), "it should be N by 3 matrix"
            pv_mesh = mesh_source
        
        if source_mesh is not None:
            mesh_data = MeshData(name=pathlib.Path(mesh_path).stem + "_mesh", 
                                path=mesh_path, 
                                source_mesh=source_mesh, 
                                pv_mesh=pv_mesh,
                                actor=None,
                                color=self.colors[self.color_counter])
            
            self.meshes[mesh_data.name] = mesh_data

            # assign a color to every mesh 
            self.color_counter += 1
            self.color_counter %= len(self.colors)

            return mesh_data

        return None

    def remove_mesh(self, name):
        del self.meshes[name]
        self.reference = None

    def render_mesh(self, camera):
        self.render.clear()
        vertices, faces = utils.get_mesh_actor_vertices_faces(self.meshes[self.reference].actor)
        mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        colors = utils.get_mesh_actor_scalars(self.meshes[self.reference].actor)
        if colors is not None: 
            assert colors.shape == vertices.shape, "colors shape should be the same as vertices shape"
            mesh = self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=1, name=self.reference)
        else:
            mesh = self.render.add_mesh(mesh_data, color=self.meshes[self.reference].color, style='surface', opacity=1, name=self.reference)
        mesh.user_matrix = self.meshes[self.reference].actor.user_matrix
        
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        image = self.render.last_image
        return image
    
    def set_scalar(self, nocs, actor_name):
        vertices, faces = utils.get_mesh_actor_vertices_faces(self.meshes[actor_name].actor)
        vertices_color = vertices
        if self.meshes[actor_name].mirror_x: vertices_color = utils.transform_vertices(vertices_color, np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        if self.meshes[actor_name].mirror_y: vertices_color = utils.transform_vertices(vertices_color, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        # get the corresponding color
        colors = utils.color_mesh(vertices_color, nocs=nocs)
        if colors.shape == vertices.shape: 
            mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
            return mesh_data, colors
        else:
            return None, None
        
    #^ Pose related 
    def current_pose(self):
        if len(self.meshes) == 1: 
            self.reference = list(self.meshes.keys())[0]
        if self.reference:
            self.meshes[self.reference].transformation_matrix = self.meshes[self.reference].actor.user_matrix
            self.meshes[self.reference].undo_poses.append(self.meshes[self.reference].actor.user_matrix)
            if len(self.meshes[self.reference].undo_poses) > 20: self.meshes[self.reference].undo_poses.pop(0)
            
    def undo_pose(self, actor_name):
        self.meshes[actor_name].transformation_matrix = self.meshes[actor_name].undo_poses.pop()
        if (self.meshes[actor_name].transformation_matrix == self.meshes[actor_name].actor.user_matrix).all():
            if len(self.meshes[actor_name].undo_poses) != 0: 
                self.meshes[actor_name].transformation_matrix = self.meshes[actor_name].undo_poses.pop()
        self.meshes[self.reference].actor.user_matrix = self.meshes[actor_name].transformation_matrix

