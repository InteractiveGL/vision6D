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
    source_mesh: trimesh.Trimesh
    pv_mesh: pv.PolyData
    actor: pv.Actor
    color: str
    spacing: List[float] = field(default_factory=[1, 1, 1])
    mirror_x: bool = False
    mirror_y: bool = False
    opacity: float = 0.3
    previous_opacity: float = 0.3
    initial_pose: np.ndarray = None
    undo_poses: List[np.ndarray] = field(default_factory=list)
    undo_vertices: List[np.ndarray] = field(default_factory=list)

class MeshStore(metaclass=Singleton):
    def __init__(self, window_size):
        self.reference: Optional[str] = None
        self.render = utils.create_render(window_size[0], window_size[1])
        self.meshes: Dict[str, MeshData] = {}
        self.color_counter = 0
        self.colors = ["cyan", "magenta", "yellow", "lime", "dodgerblue", "darkviolet", "darkorange", "darkgrey"]
        self.latlon = utils.load_latitude_longitude()
        self.toggle_anchor_mesh = True

    def reset(self): 
        self.color_counter = 0
        self.toggle_anchor_mesh = True
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
            pv_mesh = mesh_source
        
        if source_mesh is not None:
            mesh_data = MeshData(name=pathlib.Path(mesh_path).stem + "_mesh", 
                                source_mesh=source_mesh, 
                                pv_mesh=pv_mesh,
                                actor=None,
                                color=self.colors[self.color_counter],
                                spacing=[1, 1, 1])
            
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
        mesh_data = self.meshes[self.reference]
        vertices, faces = utils.get_mesh_actor_vertices_faces(mesh_data.actor)
        pv_mesh = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        colors = utils.get_mesh_actor_scalars(mesh_data.actor)
        if colors is not None: mesh = self.render.add_mesh(pv_mesh, scalars=colors, rgb=True, style='surface', opacity=1, name=self.reference)
        else: mesh = self.render.add_mesh(mesh_data, color=mesh_data.color, style='surface', opacity=1, name=self.reference)
        mesh.user_matrix = mesh_data.actor.user_matrix
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        image = self.render.last_image
        return image
    
    def get_poses_from_undo(self, mesh_data):
        transformation_matrix = mesh_data.undo_poses.pop()
        while mesh_data.undo_poses and (transformation_matrix == mesh_data.actor.user_matrix).all(): 
            transformation_matrix = mesh_data.undo_poses.pop()
        mesh_data.actor.user_matrix = transformation_matrix
        
    def get_vertices_from_undo(self, mesh_data):
        # Get the vertices and faces
        vertices, _ = utils.get_mesh_actor_vertices_faces(mesh_data.actor)
        # If there are undo vertices, get the most recent set that is different from the current vertices
        verts = mesh_data.undo_vertices.pop()  # initialize verts with current vertices
        while mesh_data.undo_vertices and (verts == vertices).all(): 
            verts = mesh_data.undo_vertices.pop()
        mesh_data.pv_mesh.points = verts
        mesh_data.actor.user_matrix = np.eye(4)
            
    def undo_actor_pose(self, name):
        mesh_data = self.meshes[name]
        if self.toggle_anchor_mesh: self.get_poses_from_undo(mesh_data)
        else: self.get_vertices_from_undo(mesh_data)
            
