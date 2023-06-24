
import numpy as np
from PyQt5 import QtWidgets
import pyvista as pv
import pathlib
import trimesh


from .. import utils
from .singleton import Singleton
from .plot_store import PlotStore
from .image_store import ImageStore

class MeshStore(metaclass=Singleton):
    def __init__(self):

        # Access other stores
        self.plot_store = PlotStore()
        self.image_store = ImageStore()

        # initialize
        self.latlon = utils.load_latitude_longitude()
        
        self.reset()

    def reset(self):
        self.reference = None
        self.mesh_path = None
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

        # Mesh pose related attributes
        self.pose_path = None
        self.transformation_matrix = np.eye(4)
        self.initial_pose = self.transformation_matrix
        self.undo_poses = {}

    def set_mesh_opacity(self, name: str, surface_opacity: float):
        assert surface_opacity>=0 and surface_opacity<=1, "mesh opacity should range from 0 to 1!"
        self.mesh_opacity[name] = surface_opacity
        self.mesh_actors[name].user_matrix = pv.array_from_vtkmatrix(self.mesh_actors[name].GetMatrix())
        self.mesh_actors[name].GetProperty().opacity = surface_opacity
        self.plot_store.add_actor(self.mesh_actors[name], pickable=True, name=name)

    def add_mesh(self, mesh_source, transformation_matrix=None):

        flag = False
                              
        if isinstance(mesh_source, pathlib.WindowsPath) or isinstance(mesh_source, str):
            self.mesh_path = str(mesh_source)
            # Load the '.mesh' file
            if pathlib.Path(mesh_source).suffix == '.mesh': mesh_source = utils.load_trimesh(mesh_source)
            else: mesh_source = pv.read(mesh_source)

        mesh_name = pathlib.Path(self.mesh_path).stem
        
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

        if not flag: self.mesh_name = None
        else:
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
            self.mesh_opacity[mesh_name] = self.surface_opacity
            mesh = self.plot_store.plotter.add_mesh(mesh_data, color=mesh_color, opacity=self.mesh_opacity[mesh_name], name=mesh_name)

            mesh.user_matrix = self.transformation_matrix if transformation_matrix is None else transformation_matrix
            self.initial_pose = mesh.user_matrix
                    
            # Add and save the actor
            actor, _ = self.plot_store.plotter.add_actor(mesh, pickable=True, name=mesh_name)

            actor_vertices, actor_faces = utils.get_mesh_actor_vertices_faces(actor)
            assert (actor_vertices == source_verts).all(), "vertices should be the same"
            assert (actor_faces == source_faces).all(), "faces should be the same"
            assert actor.name == mesh_name, "actor's name should equal to mesh_name"
            
            self.mesh_actors[mesh_name] = actor
            self.meshdict[mesh_name] = self.mesh_path
            self.mesh_opacity[mesh_name] = self.surface_opacity

            # TODO
            if self.image_store.image_path:
                self.render = utils.create_render(self.image_store.w, self.image_store.h)
            else:
                self.render = utils.create_render(self.plot_store.window_size[0], self.plot_store.window_size[1])

        return flag

    def get_mesh_colors(self, actor_name):
        vertices, _ = utils.get_mesh_actor_vertices_faces(self.mesh_actors[actor_name])
        vertices_color = vertices
        if self.plot_store.mirror_x: vertices_color = utils.transform_vertices(vertices_color, np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        if self.plot_store.mirror_y: vertices_color = utils.transform_vertices(vertices_color, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
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

    def add_pose(self, pose_source):
        if isinstance(pose_source, pathlib.WindowsPath) or isinstance(pose_source, str):
            self.pose_path = str(pose_source)
            pose_source = np.load(self.pose_path)
        self.set_transformation_matrix(pose_source)

    def set_pose(self, matrix:np.ndarray=None, rot:np.ndarray=None, trans:np.ndarray=None):
        if matrix is not None: 
            self.initial_pose = matrix
            self.reset_gt_pose()
            self.plot_store.reset_camera()
        else:
            if (rot and trans): matrix = np.vstack((np.hstack((rot, trans)), [0, 0, 0, 1]))

    def reset_gt_pose(self):
        self.register_pose(self.initial_pose)

    def current_pose(self):
        if len(self.mesh_actors) == 1: self.reference = list(self.mesh_actors.keys())[0]
        if self.reference:
            self.transformation_matrix = self.mesh_actors[self.reference].user_matrix
            self.register_pose(self.transformation_matrix)

    def update_gt_pose(self):
        self.initial_pose = self.transformation_matrix
        self.current_pose()
            
    def undo_pose(self, actor_name):
        self.transformation_matrix = self.undo_poses[actor_name].pop()
        if (self.transformation_matrix == self.mesh_actors[actor_name].user_matrix).all():
            if len(self.undo_poses[actor_name]) != 0: 
                self.transformation_matrix = self.undo_poses[actor_name].pop()

    def set_transformation_matrix(self, transformation_matrix):
        self.transformation_matrix = transformation_matrix
        if self.plot_store.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        if self.plot_store.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        self.set_pose(matrix=transformation_matrix)

    def mirror_transformation_matrix(self):
        transformation_matrix = self.transformation_matrix
        if self.plot_store.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        if self.plot_store.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        return transformation_matrix

    def register_pose(self, pose):
        for actor_name, actor in self.mesh_actors.items():
            actor.user_matrix = pose
            self.plot_store.plotter.add_actor(actor, pickable=True, name=actor_name)
