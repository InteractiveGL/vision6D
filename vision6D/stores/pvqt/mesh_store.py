
import numpy as np
from PyQt5 import QtWidgets
import pyvista as pv
import pathlib
import trimesh


from ... import utils
from ..singleton import Singleton
from .camera_store import CameraStore
from .plot_store import PlotStore

class MeshStore(metaclass=Singleton):
    def __init__(self):

        # Access other stores
        self.plot_store = PlotStore()
        self.camera_store = CameraStore()

        # initialize
        self.latlon = utils.load_latitude_longitude()
        self.reference = None
        self.mesh_actors = {}
        self.meshdict = {}
        
        self.colors = ["cyan", "magenta", "yellow", "lime", "dodgerblue", "darkviolet", "darkorange", "forestgreen"]
        self.used_colors = []
        self.mesh_colors = {}

        self.surface_opacity = 0.3
        self.mesh_opacity = {}
        self.store_mesh_opacity = {}
        
        # Set mesh spacing
        self.mesh_spacing = [1, 1, 1]

    def set_mesh_opacity(self, name: str, surface_opacity: float):
        assert surface_opacity>=0 and surface_opacity<=1, "mesh opacity should range from 0 to 1!"
        self.mesh_opacity[name] = surface_opacity
        self.mesh_actors[name].user_matrix = pv.array_from_vtkmatrix(self.mesh_actors[name].GetMatrix())
        self.mesh_actors[name].GetProperty().opacity = surface_opacity
        self.plot_store.add_actor(self.mesh_actors[name], pickable=True, name=name)

    def add_mesh(self, mesh_name, mesh_source, transformation_matrix=None):

        flag = False
                              
        if isinstance(mesh_source, pathlib.WindowsPath) or isinstance(mesh_source, str):
            # Load the '.mesh' file
            if pathlib.Path(mesh_source).suffix == '.mesh': mesh_source = utils.load_trimesh(mesh_source)
            # Load the '.ply' file
            else: mesh_source = pv.read(mesh_source)

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

        if not flag:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "The mesh format is not supported!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

        # consider the mesh verts spacing
        mesh_data.points = mesh_data.points * self.mesh_spacing

        # assign a color to every mesh
        if len(self.colors) != 0: mesh_color = self.colors.pop(0)
        else:
            self.colors = self.used_colors
            mesh_color = self.colors.pop(0)
            self.used_colors = []

        self.used_colors.append(mesh_color)
        self.mesh_colors[mesh_name] = mesh_color
        self.qt_store.color_button.setText(self.mesh_colors[mesh_name])
        mesh = self.plotter.add_mesh(mesh_data, color=mesh_color, opacity=self.mesh_opacity[mesh_name], name=mesh_name)

        mesh.user_matrix = self.transformation_matrix if transformation_matrix is None else transformation_matrix
        self.initial_pose = mesh.user_matrix
                
        # Add and save the actor
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=mesh_name)

        actor_vertices, actor_faces = utils.get_mesh_actor_vertices_faces(actor)
        assert (actor_vertices == source_verts).all(), "vertices should be the same"
        assert (actor_faces == source_faces).all(), "faces should be the same"
        assert actor.name == mesh_name, "actor's name should equal to mesh_name"
        
        self.mesh_actors[mesh_name] = actor

    def get_mesh_colors(self, actor_name):
        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_actors[actor_name])
        vertices_color = vertices
        if self.camera_store.mirror_x: vertices_color = utils.transform_vertices(vertices_color, np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        if self.camera_store.mirror_y: vertices_color = utils.transform_vertices(vertices_color, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        return vertices_color, vertices

    def set_mesh_colors(self, actor_name, colors):
        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_actors[actor_name])
        mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))

        mesh = self.plot_store.plotter.add_mesh(mesh_data, scalars=colors, rgb=True, opacity=self.mesh_opacity[actor_name], name=actor_name)
        transformation_matrix = pv.array_from_vtkmatrix(self.mesh_actors[actor_name].GetMatrix())
        mesh.user_matrix = transformation_matrix
        actor, _ = self.plot_store.plotter.add_actor(mesh, pickable=True, name=actor_name)
        actor_colors = utils.get_mesh_actor_scalars(actor)

        assert (actor_colors == colors).all(), "actor_colors should be the same as colors"
        assert actor.name == actor_name, "actor's name should equal to actor_name"
        self.mesh_actors[actor_name] = actor

    def remove_actor(self, name):
        if name in self.mesh_actors:
            mesh_actor = self.mesh_actors[name]
            del self.mesh_actors[name] # remove the item from the mesh dictionary
            del self.mesh_colors[name]
            del self.mesh_opacity[name]
            del self.meshdict[name]
            self.reference = None
            self.mesh_spacing = [1, 1, 1]
            return mesh_actor
        return None
