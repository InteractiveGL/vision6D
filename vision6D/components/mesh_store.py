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

import trimesh
import pyvista as pv
import numpy as np

from . import Singleton
from ..tools import utils

# contains mesh objects

class MeshStore(metaclass=Singleton):
    def __init__(self, window_size):
        self.reset()
        self.render = utils.create_render(window_size[0], window_size[1])

    def reset(self):
        self.reference = None
        self.mesh_path = None
        self.mesh_name = None
        self.mirror_x = False
        self.mirror_y = False
        self.mesh_actors = {}
        self.meshdict = {}
        self.latlon = utils.load_latitude_longitude()
        
        self.colors = ["cyan", "magenta", "yellow", "lime", "dodgerblue", "darkviolet", "darkorange", "darkgrey"]
        self.used_colors = []
        self.mesh_colors = {}

        self.surface_opacity = 0.3
        self.mesh_opacity = {}
        self.store_mesh_opacity = {}
        
        # Set mesh spacing
        self.mesh_spacing = [1, 1, 1]

        # Pose as meshes' attributes
        self.pose_path = None
        self.transformation_matrix = np.eye(4)
        self.initial_pose = None
        self.undo_poses = {}

    #^ Mesh related
    def add_mesh(self, mesh_source):
        flag = False
                              
        if isinstance(mesh_source, pathlib.WindowsPath) or isinstance(mesh_source, str):
            self.mesh_path = str(mesh_source)
            if pathlib.Path(mesh_source).suffix == '.mesh': mesh_source = utils.load_trimesh(mesh_source)
            else: mesh_source = pv.read(mesh_source)

        self.mesh_name = pathlib.Path(self.mesh_path).stem
        self.meshdict[self.mesh_name] = self.mesh_path
        self.mesh_opacity[self.mesh_name] = self.surface_opacity

        if isinstance(mesh_source, trimesh.Trimesh):
            assert (mesh_source.vertices.shape[1] == 3 and mesh_source.faces.shape[1] == 3), "it should be N by 3 matrix"
            mesh_data = pv.wrap(mesh_source)
            source_verts = mesh_source.vertices * self.mesh_spacing
            source_faces = mesh_source.faces
            flag = True

        if isinstance(mesh_source, pv.PolyData):
            mesh_data = mesh_source
            source_verts = mesh_source.points * self.mesh_spacing
            source_faces = mesh_source.faces.reshape((-1, 4))[:, 1:]
            flag = True

        if flag:
            # consider the mesh verts spacing
            mesh_data.points = mesh_data.points * self.mesh_spacing
            if self.initial_pose is None: self.initial_pose = self.transformation_matrix

            # assign a color to every mesh
            if len(self.colors) != 0: 
                mesh_color = self.colors.pop(0)
            else:
                self.colors = self.used_colors
                mesh_color = self.colors.pop(0)
                self.used_colors = []

            self.used_colors.append(mesh_color)
            # store neccessary attributes
            self.mesh_colors[self.mesh_name] = mesh_color
                        
            return mesh_data, source_verts, source_faces
        else:
            self.reset()
            return None, None, None

    def remove_mesh(self, name):
        del self.mesh_actors[name] # remove the item from the mesh dictionary
        del self.mesh_colors[name]
        del self.mesh_opacity[name]
        del self.meshdict[name]
        self.reference = None

    def render_mesh(self, camera):
        self.render.clear()
        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_actors[self.reference])
        mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        colors = utils.get_mesh_actor_scalars(self.mesh_actors[self.reference])
        if colors is not None: 
            assert colors.shape == vertices.shape, "colors shape should be the same as vertices shape"
            mesh = self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=1, name=self.reference)
        else:
            mesh = self.render.add_mesh(mesh_data, color=self.mesh_colors[self.reference], style='surface', opacity=1, name=self.reference)
        mesh.user_matrix = self.mesh_actors[self.reference].user_matrix
        
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        image = self.render.last_image
        return image
    
    def set_scalar(self, nocs, actor_name):
        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_actors[actor_name])
        vertices_color = vertices
        if self.mirror_x: vertices_color = utils.transform_vertices(vertices_color, np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        if self.mirror_y: vertices_color = utils.transform_vertices(vertices_color, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        # get the corresponding color
        colors = utils.color_mesh(vertices_color, nocs=nocs)
        if colors.shape == vertices.shape: 
            mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
            return mesh_data, colors
        else:
            return None, None
        
    #^ Pose related 
    def current_pose(self):
        if len(self.mesh_actors) == 1: 
            self.reference = list(self.mesh_actors.keys())[0]
        if self.reference:
            self.transformation_matrix = self.mesh_actors[self.reference].user_matrix
            # save the previous poses to self.undo_poses      
            if self.reference not in self.undo_poses: self.undo_poses[self.reference] = []
            self.undo_poses[self.reference].append(self.mesh_actors[self.reference].user_matrix)
            if len(self.undo_poses[self.reference]) > 20: self.undo_poses[self.reference].pop(0)
            
    def undo_pose(self, actor_name):
        self.transformation_matrix = self.undo_poses[actor_name].pop()
        if (self.transformation_matrix == self.mesh_actors[actor_name].user_matrix).all():
            if len(self.undo_poses[actor_name]) != 0: 
                self.transformation_matrix = self.undo_poses[actor_name].pop()
        self.mesh_actors[actor_name].user_matrix = self.transformation_matrix

