from typing import List, Dict
import pathlib
import logging
import numpy as np
from PIL import Image
import copy
import types
import functools

import pyvista as pv
import trimesh
import cv2
from easydict import EasyDict
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyvista.plotting.render_passes import RenderPasses

logger = logging.getLogger("vision6D")

def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g

def fread(fid, _len, _type):
    if _len == 0:
        return np.empty(0)
    if _type == "int16":
        _type = np.int16
    elif _type == "int32":
        _type = np.int32
    elif _type == "float":
        _type = np.float32
    elif _type == "double":
        _type = np.double
    elif _type == "char":
        _type = np.byte
    elif _type == "uint8":
        _type = np.uint8
    else:
        raise NotImplementedError(f"Invalid _type: {_type}")

    return np.fromfile(fid, _type, _len)

def meshread(fid, linesread=False, meshread2=False):
    """Reads mesh from fid data stream

    Parameters
    ----------
    fid (io.BufferedStream)
        Input IO stream
    _type (str, optional):
        Specifying the data _type for the last fread
    linesread (bool, optional)
        Distinguishing different use cases,
            False => meshread (default)
            True  => linesread

    """

    # Creating mesh instance
    mesh = EasyDict()

    # Reading parameters for mesh
    mesh.id = fread(fid, 1, "int32")
    mesh.numverts = fread(fid, 1, "int32")[0]
    mesh.numtris = fread(fid, 1, "int32")[0]

    # Loading mesh data
    n = fread(fid, 1, "int32")
    if n == -1:
        mesh.orient = fread(fid, 3, "int32")
        mesh.dim = fread(fid, 3, "int32")
        mesh.sz = fread(fid, 3, "float")
        mesh.color = fread(fid, 3, "int32")
    else:
        mesh.color = np.zeros(3)
        mesh.color[0] = n
        mesh.color[1:3] = fread(fid, 2, "int32")

    # Given input parameter `linesread`
    if linesread:
        mesh.vertices = fread(fid, 3 * mesh.numverts, "float").reshape(
            [3, mesh.numverts], order="F"
        )
        mesh.triangles = fread(fid, 2 * mesh.numtris, "int32").reshape(
            [2, mesh.numtris], order="F"
        )
    # Given input parameter `meshread2`
    elif meshread2:
        mesh.vertices = fread(fid, 3 * mesh.numverts, "double").reshape(
            [3, mesh.numverts], order="F"
        )
        mesh.triangles = fread(fid, 3 * mesh.numtris, "int32").reshape(
            [3, mesh.numtris], order="F"
        )
    # Given input parameter `meshread`
    else:
        # Loading mesh vertices and triangles
        mesh.vertices = fread(fid, 3 * mesh.numverts, "float").reshape(
            [3, mesh.numverts], order="F"
        )
        mesh.triangles = fread(fid, 3 * mesh.numtris, "int32").reshape(
            [3, mesh.numtris], order="F"
        )

    # Return data
    return mesh

class App:

    def __init__(self, register, image_path, scale_factor=[1,1,1]):
        
        self.register = register
        self.reference = None
        self.image_actors = {}
        self.mesh_actors = {}
        
        self.image_polydata = {
            'image': pv.read(image_path),
            'image-origin': None
        }
        
        self.mesh_polydata = {}
        self.binded_meshes = {}
        
        self.image_polydata['image'] = self.image_polydata['image'].scale(scale_factor, inplace=False)
        self.image_polydata["image-origin"] = self.image_polydata['image'].copy()

        # "xy" camera view
        self.xyviewcamera = pv.Camera()
        self.xyviewcamera.focal_point = (9.6, 5.4, 0)
        self.xyviewcamera.up = (0.0, 1.0, 0.0)
        
        self.transformation_matrix = np.array(
            [[-0.66487539, -0.21262585, -0.71605235,  3.25029551],
            [ 0.08437209, -0.97387229,  0.21084143, 30.99098483],
            [-0.74217388,  0.07976845,  0.66544341, 14.47777792],
            [ 0.        ,  0.        ,  0.        ,  1.        ]])
        
        # self.transformation_matrix = np.array([[1, 0, 0, 0],
        #                                     [0, 1, 0, 0],
        #                                     [0, 0, 1, 0],
        #                                     [0, 0, 0, 1]])
                
        if self.register:
            self.pl = pv.Plotter(window_size=[1920, 1080])
        else:
            self.pl = pv.Plotter(window_size=[1920, 1080], off_screen=True)
            self.pl.store_image = True
            
        self.xyviewcamera.position = (9.6, 5.4, 20)
        
        # render ossicles
        self.pr = pv.Plotter(window_size=[1920, 1080], lighting=None, off_screen=True)
        self.pr.store_image = True
        
        # render RGB image
        self.pi = pv.Plotter(window_size=[1920, 1080], lighting=None, off_screen=True)
        self.pi.store_image = True
        
        self.load_image(image_path, scale_factor)
        
    def set_reference(self, name:str):
        self.reference = name
        
    def bind_meshes(self, main_mesh: str, key: str, other_meshes: List[str]):
        self.binded_meshes[main_mesh] = {'key': key, 'meshes': other_meshes}
        
    def load_image(self, image_path:pathlib.Path, scale_factor:list=[1,1,1]):

        # Then add it to the plotter
        image = self.pl.add_mesh(self.image_polydata['image'], rgb=True, opacity=0.35, name='image')
        actor, _ = self.pl.add_actor(image, name="image")

        # Save actor for later
        self.image_actors["image"] = actor
        self.image_actors["image-origin"] = actor.copy()
        
    def transform_vertices(self, transformation_matrix, vertices):
        
        transformation_matrix = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
        
        ones = np.ones((vertices.shape[0], 1))
        homogeneous_vertices = np.append(vertices, ones, axis=1)
        transformed_vertices = (transformation_matrix @ homogeneous_vertices.T)[:3].T
        
        return transformed_vertices

    def color_mesh(self, vertices):
        colors = copy.deepcopy(vertices)
        # normalize vertices and center it to 0
        colors[0] = (vertices[0] - np.min(vertices[0])) / (np.max(vertices[0]) - np.min(vertices[0])) - 0.5
        colors[1] = (vertices[1] - np.min(vertices[1])) / (np.max(vertices[1]) - np.min(vertices[1])) - 0.5
        colors[2] = (vertices[2] - np.min(vertices[2])) / (np.max(vertices[2]) - np.min(vertices[2])) - 0.5
        colors = colors.T + np.array([0.5, 0.5, 0.5])
        
        return colors

    def load_meshes(self, paths: Dict[str, pathlib.Path]):
        
        for mesh_name, mesh_path in paths.items():
            
            # Load the mesh
            mesh_data = pv.read(mesh_path)
            self.mesh_polydata[mesh_name] = mesh_data
            
            # Apply transformation to the mesh vertices
            transformed_points = self.transform_vertices(self.transformation_matrix, mesh_data.points)
            colors = self.color_mesh(transformed_points.T)
            
            # Color the vertex
            mesh_data.point_data.set_scalars(colors)

            mesh = self.pl.add_mesh(mesh_data, rgb=True, show_scalar_bar=False, name=mesh_name)
            
            mesh.user_matrix = self.transformation_matrix
            
            actor, _ = self.pl.add_actor(mesh, name=mesh_name)
            
            # Save actor for later
            self.mesh_actors[mesh_name] = actor
            
            logger.debug(f"\n{mesh_name} orientation: {self.mesh_actors[mesh_name].orientation}")
            logger.debug(f"\n{mesh_name} position: {self.mesh_actors[mesh_name].position}")
            
    def event_zoom_out(self, *args):
        self.pl.camera.zoom(0.5)
        logger.debug("event_zoom_out callback complete")

    def event_reset_camera(self, *args):
        self.pl.camera = self.xyviewcamera.copy()
        logger.debug("reset_camera_event callback complete")

    def event_reset_image_position(self, *args):
        self.image_actors["image"] = self.image_actors["image-origin"].copy() # have to use deepcopy to prevent change self.image_actors["image-origin"] content
        self.pl.add_actor(self.image_actors["image"], name="image")
        logger.debug("reset_image_position callback complete")

    def event_track_registration(self, *args):

        self.event_realign_meshes(main_mesh="ossicles", other_meshes=['facial_nerve', 'chorda'])
        self.event_change_color()
        for actor_name, actor in self.mesh_actors.items():
            logger.debug(f"<Actor {actor_name}> RT: \n{actor.user_matrix}")
    
    def event_realign_meshes(self, *args, main_mesh=None, other_meshes=[]):
        
        objs = {'fix' : main_mesh,
                'move': other_meshes}
        
        transformation_matrix = self.mesh_actors[f"{objs['fix']}"].user_matrix
        
        for obj in objs['move']:
            self.mesh_actors[f"{obj}"].user_matrix = transformation_matrix
        
        logger.debug(f"realign: main => {main_mesh}, others => {other_meshes} complete")
        
    def event_gt_position(self, *args):
        
        for _, actor in self.mesh_actors.items():
            actor.user_matrix = self.transformation_matrix
        
        self.event_change_color()

        logger.debug(f"\ncurrent gt rt: \n{self.transformation_matrix}")
        logger.debug("event_gt_position callback complete")
        
    def event_change_gt_position(self, *args):
        if self.reference:
            self.transformation_matrix = self.mesh_actors[self.reference].user_matrix
            for _, actor in self.mesh_actors.items():
                actor.user_matrix = self.transformation_matrix
                
            self.event_change_color()
            
            logger.debug(f"\ncurrent gt rt: \n{self.transformation_matrix}")
            logger.debug("event_change_gt_position callback complete")
        else:
            logger.error("reference not set")
        
    def event_change_color(self, *args):
        
        if self.reference:
            transformation_matrix = self.mesh_actors[self.reference].user_matrix
            container = self.mesh_actors.copy()
            
            for actor_name, actor in self.mesh_actors.items():
                
                # Color the vertex
                transformed_points = self.transform_vertices(transformation_matrix, self.mesh_polydata[f'{actor_name}'].points)
                colors = self.color_mesh(transformed_points.T)
                self.mesh_polydata[f'{actor_name}'].point_data.set_scalars(colors)
                
                mesh = self.pr.add_mesh(self.mesh_polydata[f'{actor_name}'], rgb=True, show_scalar_bar=False)
                mesh.user_matrix = transformation_matrix
                
                actor, _ = self.pl.add_actor(mesh, name=actor_name)
                
                # Save the new actor to a container
                container[actor_name] = actor

            self.mesh_actors = container
            
            logger.debug("event_change_color callback complete")
        else:
            logger.error("reference not set")
    
    def plot(self):

        self.pl.enable_joystick_actor_style()

        # Register callbacks
        self.pl.add_key_event('c', self.event_reset_camera)
        self.pl.add_key_event('z', self.event_zoom_out)
        self.pl.add_key_event('d', self.event_reset_image_position)
        self.pl.add_key_event('t', self.event_track_registration)
        
        # self.pl.add_key_event('g', self.event_realign_facial_nerve_chorda)
        # self.pl.add_key_event('h', self.event_realign_facial_nerve_ossicles)
        # self.pl.add_key_event('j', self.event_realign_chorda_ossicles)

        
        logger.debug(self.binded_meshes)
        for main_mesh, mesh_data in self.binded_meshes.items():
            event_func = functools.partial(self.event_realign_meshes, main_mesh=main_mesh, other_meshes=mesh_data['meshes'])
            self.pl.add_key_event(mesh_data['key'], event_func)
        
        self.pl.add_key_event('k', self.event_gt_position)
        self.pl.add_key_event('l', self.event_change_gt_position)
        self.pl.add_key_event('v', self.event_change_color)
        
        # Set the camera initial parameters
        self.pl.camera = self.xyviewcamera.copy()
        
        if self.register:
            self.pl.add_axes()
            # add the camera orientation to move the camera
            _ = self.pl.add_camera_orientation_widget()
            # Actual presenting
            cpos = self.pl.show(title="vision6D", return_cpos=True)
        else:
            self.pl.disable()
            cpos = self.pl.show(title="vision6D", return_cpos=True)
            result = self.pl.last_image
            res_plot = Image.fromarray(result)
            res_plot.save("res_plot.png")
        
        # logger.debug(f"\nrt: \n{self.mesh_actors['ossicles'].user_matrix}")
        logger.debug(f"\ncpos: {cpos}")
        
    def render_image(self, image_path, scale_factor):
        self.pi.enable_joystick_actor_style()
        image = pv.read(image_path)
        image = image.scale(scale_factor, inplace=False)
        image = self.pi.add_mesh(image, rgb=True, opacity=1)
        self.pi.camera = self.xyviewcamera.copy()
        self.pi.disable()
        self.pi.show()
        result = self.pi.last_image
        res_render = Image.fromarray(result)
        res_render.save("image.png")
        print("hhh")

    def render_ossicles(self, scale_factor, mesh_path):
    
        self.pr.enable_joystick_actor_style()
        self.pr.set_background('white')
        
        image = pv.read("black_background.jpg")
        image = image.scale(scale_factor, inplace=False)
        # generate white image
        image = self.pr.add_mesh(image, rgb=True, opacity=0, show_scalar_bar=False)
        # # generate grey image
        # image = self.pr.add_mesh(image, rgb=True, opacity=0.5, show_scalar_bar=False)
        mesh = pv.read(mesh_path)
        transformed_points = self.transform_vertices(self.transformation_matrix, mesh.points)
        colors = self.color_mesh(transformed_points.T)
        # Color the vertex
        mesh.point_data.set_scalars(colors)
        mesh = self.pr.add_mesh(mesh, rgb=True, show_scalar_bar=False)
        
        mesh.user_matrix = self.transformation_matrix
    
        self.pr.camera = self.xyviewcamera.copy()
        
        self.pr.disable()
        self.pr.show()
        result = self.pr.last_image
        res_render = Image.fromarray(result)
        
        if image.prop.opacity == 0:
            res_render.save("res_render.png")
        elif image.prop.opacity == 0.5:
            res_render.save("res_render_grey.png")
        print("hhh")
        
    def plot_render(self, scale_factor, render_path):
        self.plot()
        self.render_ossicles(scale_factor, render_path, rgb=True)