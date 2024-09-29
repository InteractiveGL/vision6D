'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: mesh_container.py
@time: 2023-07-03 20:27
@desc: create container for mesh related actions in application
'''
import pathlib
import trimesh
import matplotlib
import numpy as np
import pyvista as pv
from typing import Dict

from PyQt5 import QtWidgets

from ..tools import utils
from ..components import MeshModel
from .singleton import Singleton

class MeshContainer(metaclass=Singleton):
    def __init__(self, plotter):
        self.plotter = plotter
        self.reference = None
        self.meshes: Dict[str, MeshModel] = {}
        self.colors = ["wheat", "cyan", "magenta", "yellow", "lime", "dodgerblue", "white", "black"]
        self.mesh_model = MeshModel()

    def reset(self):
        pass
        
    def add_mesh(self, mesh_source, transformation_matrix, w, h):
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
            self.mesh_model.name = name
            self.mesh_model.path = mesh_path
            self.mesh_model.source_obj = source_mesh
            self.mesh_model.pv_obj = pv_mesh
            
            # set spacing for the mesh
            self.mesh_model.pv_obj.points *= self.mesh_model.spacing

            self.meshes[self.mesh_model.name] = self.mesh_model
            self.mesh_model.color = self.colors[self.meshes.index(self.mesh_model) % len(self.colors)]
        
            """ add a mesh to the pyqt frame """
            mesh = self.plotter.add_mesh(self.mesh_model.pv_obj, color=self.mesh_model.color, opacity=self.mesh_model.opacity, name=self.mesh_model.name)
            mesh.user_matrix = transformation_matrix
            actor, _ = self.plotter.add_actor(mesh, pickable=True, name=self.mesh_model.name)
            self.mesh_model.actor = actor
            self.reference = self.mesh_model.name
            return self.mesh_model
        
    def set_color(self, color, name):
        scalars = None
        mesh_model = self.meshes[name]
        if color == "nocs": scalars = utils.color_mesh_nocs(mesh_model.pv_obj.points)
        elif color == "texture":
            texture_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.npy)")
            if texture_path: 
                mesh_model.texture_path = texture_path
                scalars = np.load(texture_path) / 255 # make sure the color range is from 0 to 1

        if scalars is not None: 
            mesh = self.plotter.add_mesh(mesh_model.pv_obj, scalars=scalars, rgb=True, opacity=mesh_model.opacity, name=name)
        else:
            mesh = self.plotter.add_mesh(mesh_model.pv_obj, opacity=mesh_model.opacity, name=name)
            mesh.GetMapper().SetScalarVisibility(0)
            if color in self.colors: mesh.GetProperty().SetColor(matplotlib.colors.to_rgb(color))
            else:
                mesh_model.color = "wheat"
                mesh.GetProperty().SetColor(matplotlib.colors.to_rgb(mesh_model.color))

        mesh.user_matrix = mesh_model.actor.user_matrix
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=name)
        mesh_model.actor = actor #^ very import to change the actor too!
        if scalars is None and color == 'texture': color = mesh_model.color
        return color
            
    def set_mesh_opacity(self, name: str, mesh_opacity: float):
        mesh_model = self.meshes[name]
        mesh_model.previous_opacity = mesh_model.opacity
        mesh_model.opacity = mesh_opacity
        mesh_model.actor.user_matrix = pv.array_from_vtkmatrix(mesh_model.actor.GetMatrix())
        mesh_model.actor.GetProperty().opacity = mesh_opacity

    def remove_mesh(self, name):
        del self.meshes[name]
        self.reference = None
    
    def get_poses_from_undo(self, name):
        transformation_matrix = self.meshes[name].undo_poses.pop()
        while self.meshes[name].undo_poses and (transformation_matrix == self.meshes[name].actor.user_matrix).all(): 
            transformation_matrix = self.meshes[name].undo_poses.pop()
        self.meshes[name].actor.user_matrix = transformation_matrix
            
    def undo_actor_pose(self, name):
        mesh_model = self.meshes[name]
        self.get_poses_from_undo(mesh_model)

    def render_mesh(self, name, camera, width, height):
        render = utils.create_render(width, height); render.clear()
        mesh_model = self.meshes[name]
        vertices, faces = utils.get_mesh_actor_vertices_faces(mesh_model.actor)
        pv_mesh = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        colors = utils.get_mesh_actor_scalars(mesh_model.actor)
        if colors is not None: mesh = render.add_mesh(pv_mesh, scalars=colors, rgb=True, style='surface', opacity=1, name=name)
        else: mesh = render.add_mesh(pv_mesh, color=mesh_model.color, style='surface', opacity=1, name=name)
        mesh.user_matrix = mesh_model.actor.user_matrix
        # set the light source to add the textures
        light = pv.Light(light_type='headlight')
        render.add_light(light)
        render.camera = camera
        render.disable()
        render.show(auto_close=False)
        image = render.last_image
        return image