'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: mesh_container.py
@time: 2023-07-03 20:27
@desc: create container for mesh related actions in application
'''
import os
import pathlib

import trimesh
import PIL.Image
import matplotlib
import numpy as np
import pyvista as pv

from PyQt5 import QtWidgets

from ..tools import utils
from ..components import MaskStore
from ..components import ImageStore
from ..components import MeshStore

from ..path import PKG_ROOT

class MeshContainer:
    def __init__(self, plotter):

        self.plotter = plotter
        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.mesh_store = MeshStore()
        
    def add_mesh(self, mesh_data, transformation_matrix):
        """ add a mesh to the pyqt frame """
        mesh = self.plotter.add_mesh(mesh_data.pv_mesh, color=mesh_data.color, opacity=mesh_data.opacity, name=mesh_data.name)
        mesh.user_matrix = transformation_matrix
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=mesh_data.name)
        mesh_data.actor = actor
        return mesh_data
        
    def set_color(self, color, name):
        scalars = None
        mesh_data = self.mesh_store.meshes[name]
        if color == "nocs": scalars = utils.color_mesh_nocs(mesh_data.pv_mesh.points)
        elif color == "texture":
            texture_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.npy)")
            if texture_path: 
                mesh_data.texture_path = texture_path
                scalars = np.load(texture_path) / 255 # make sure the color range is from 0 to 1

        if scalars is not None: mesh = self.plotter.add_mesh(mesh_data.pv_mesh, scalars=scalars, rgb=True, opacity=mesh_data.opacity, name=name)
        else:
            mesh = self.plotter.add_mesh(mesh_data.pv_mesh, opacity=mesh_data.opacity, name=name)
            mesh.GetMapper().SetScalarVisibility(0)
            if color in self.mesh_store.colors: mesh.GetProperty().SetColor(matplotlib.colors.to_rgb(color))
            else:
                mesh_data.color = "wheat"
                mesh.GetProperty().SetColor(matplotlib.colors.to_rgb(mesh_data.color))

        mesh.user_matrix = mesh_data.actor.user_matrix
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=name)
        mesh_data.actor = actor #^ very import to change the actor too!
        if scalars is None and color == 'texture': color = mesh_data.color
        return color
            
    def set_mesh_opacity(self, name: str, mesh_opacity: float):
        mesh_data = self.mesh_store.meshes[name]
        mesh_data.previous_opacity = mesh_data.opacity
        mesh_data.opacity = mesh_opacity
        mesh_data.actor.user_matrix = pv.array_from_vtkmatrix(mesh_data.actor.GetMatrix())
        mesh_data.actor.GetProperty().opacity = mesh_opacity

    def render_mesh(self, camera):
        render = utils.create_render(self.image_store.width, self.image_store.height); render.clear()
        mesh_data = self.mesh_store.meshes[self.mesh_store.reference]
        vertices, faces = utils.get_mesh_actor_vertices_faces(mesh_data.actor)
        pv_mesh = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        colors = utils.get_mesh_actor_scalars(mesh_data.actor)
        if colors is not None: mesh = render.add_mesh(pv_mesh, scalars=colors, rgb=True, style='surface', opacity=1, name=self.mesh_store.reference)
        else: mesh = render.add_mesh(pv_mesh, color=mesh_data.color, style='surface', opacity=1, name=self.mesh_store.reference)
        mesh.user_matrix = mesh_data.actor.user_matrix
        # set the light source to add the textures
        light = pv.Light(light_type='headlight')
        render.add_light(light)
        render.camera = camera
        render.disable()
        render.show(auto_close=False)
        image = render.last_image
        return image