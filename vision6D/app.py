from typing import List, Dict, Tuple
import pathlib
import logging
import numpy as np
from PIL import Image
import copy
import math
import functools

import pyvista as pv
import trimesh
import cv2
from easydict import EasyDict
import matplotlib.pyplot as plt

from . import utils

logger = logging.getLogger("vision6D")

class App:

    def __init__(
            self, 
            register, 
            focal_length:int=20, 
            height:int=1920, 
            width:int=1080, 
            scale:float=1/100
        ):
        
        self.register = register
        self.reference = None
        
        self.image_actors = {}
        self.mesh_actors = {}
        
        self.image_polydata = {}
        self.mesh_polydata = {}
        
        self.binded_meshes = {}
        
        # "xy" camera view
        self.xyviewcamera = pv.Camera()
        self.xyviewcamera.up = (0.0, 1.0, 0.0)
        self.set_camera_intrinsics(focal_length, height, width, scale)       
        
        # w = 19.20
        # h = 10.80
    
        # cx = intrinsic[0,2]
        # cy = intrinsic[1,2]
        # f = intrinsic[0,0]
        
        # wcx = -2*(cx - float(w)/2) / w
        # wcy =  2*(cy - float(h)/2) / h
        
        # self.xyviewcamera.SetWindowCenter(wcx, wcy)
        # view_angle = 180 / math.pi * (2.0 * math.atan2(h/2.0, f))
        # self.xyviewcamera.SetViewAngle(view_angle)

        self.transformation_matrix = np.array(
            [[-0.66487539, -0.21262585, -0.71605235,  3.25029551],
            [ 0.08437209, -0.97387229,  0.21084143, 30.99098483],
            [-0.74217388,  0.07976845,  0.66544341, 14.47777792],
            [ 0.        ,  0.        ,  0.        ,  1.        ]])
        
        # self.transformation_matrix = np.eye(4)
        
    #     self.transformation_matrix = np.array([[-0.00017772, -0.99999998,  0.00000097,  1.70417365],
    #    [ 0.99999998, -0.00017772, -0.00000393,  1.64181747],
    #    [ 0.00000393,  0.00000097,  1.        , -0.98884161],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
        
                
        if self.register:
            self.pv_plotter = pv.Plotter(window_size=[height, width])
        else:
            self.pv_plotter = pv.Plotter(window_size=[height, width], off_screen=True)
            self.pv_plotter.store_image = True
        
        # render ossicles
        self.pv_render = pv.Plotter(window_size=[height, width], lighting=None, off_screen=True)
        self.pv_render.store_image = True
        
        # render RGB image
        self.pv_render_image = pv.Plotter(window_size=[height, width], lighting=None, off_screen=True)
        self.pv_render_image.store_image = True
        
    def set_camera_intrinsics(
            self, 
            focal_length:int, 
            height:int,
            width:int,
            scale:float,
            offsets:Tuple[int]=[0,0]
        ):
        
        scaled_height = height*scale
        scaled_width = width*scale
        principle_points = (scaled_height/2+offsets[0]*scale, scaled_width/2+offsets[1]*scale)
        
        self.xyviewcamera.focal_point = (*principle_points, 0)
        self.xyviewcamera.position = (*principle_points, focal_length)
             
        vmtx = self.xyviewcamera.GetModelViewTransformMatrix()
        mtx = pv.array_from_vtkmatrix(vmtx)
        
        model_transform_matrix = np.linalg.inv([[1, 0, 0, principle_points[0]],
                                            [0, 1, 0, principle_points[1]],
                                            [0, 0, 1, focal_length],
                                            [0, 0, 0, 1]])
        
        assert (mtx == model_transform_matrix).all(), "the two matrix should be equal"
        
        # Set camera intrinsic attribute
        self.camera_intrinsics = np.array([
            [focal_length, 0, principle_points[0]],
            [0, focal_length, principle_points[1]],
            [0, 0, 0]
        ])
        
    def set_reference(self, name:str):
        self.reference = name
        
    def bind_meshes(self, main_mesh: str, key: str, other_meshes: List[str]):
        self.binded_meshes[main_mesh] = {'key': key, 'meshes': other_meshes}
        
    def load_image(self, image_path:pathlib.Path, scale_factor:list=[1,1,1]):
        
        self.image_polydata['image'] = pv.read(image_path)
        self.image_polydata['image'] = self.image_polydata['image'].scale(scale_factor, inplace=False)
        self.image_polydata["image-origin"] = self.image_polydata['image'].copy()

        # Then add it to the plotter
        image = self.pv_plotter.add_mesh(self.image_polydata['image'], rgb=True, opacity=0.35, name='image')
        actor, _ = self.pv_plotter.add_actor(image, name="image")

        # Save actor for later
        self.image_actors["image"] = actor
        self.image_actors["image-origin"] = actor.copy()

    def load_meshes(self, paths: Dict[str, (pathlib.Path or pv.PolyData)]):
        
        for mesh_name, mesh_source in paths.items():
            
            if isinstance(mesh_source, pathlib.WindowsPath):
                # Load the mesh
                if '.ply' in str(mesh_source):
                    mesh_data = pv.read(mesh_source)
                elif '.mesh' in str(mesh_source):
                    mesh_data = pv.wrap(utils.load_trimesh(mesh_source))
            elif isinstance(mesh_source, pv.PolyData):
                mesh_data = mesh_source
                
            self.mesh_polydata[mesh_name] = mesh_data
            # Apply transformation to the mesh vertices
            transformed_points = utils.transform_vertices(self.transformation_matrix, mesh_data.points)
            colors = utils.color_mesh(transformed_points.T)
            
            # Color the vertex
            mesh_data.point_data.set_scalars(colors)

            mesh = self.pv_plotter.add_mesh(mesh_data, rgb=True, name=mesh_name)
            
            mesh.user_matrix = self.transformation_matrix
            
            actor, _ = self.pv_plotter.add_actor(mesh, name=mesh_name)
            
            # Save actor for later
            self.mesh_actors[mesh_name] = actor
            
            logger.debug(f"\n{mesh_name} orientation: {self.mesh_actors[mesh_name].orientation}")
            logger.debug(f"\n{mesh_name} position: {self.mesh_actors[mesh_name].position}")
            
    def event_zoom_out(self, *args):
        self.pv_plotter.camera.zoom(0.5)
        logger.debug("event_zoom_out callback complete")

    def event_reset_camera(self, *args):
        self.pv_plotter.camera = self.xyviewcamera.copy()
        logger.debug("reset_camera_event callback complete")

    def event_reset_image_position(self, *args):
        self.image_actors["image"] = self.image_actors["image-origin"].copy() # have to use deepcopy to prevent change self.image_actors["image-origin"] content
        self.pv_plotter.add_actor(self.image_actors["image"], name="image")
        logger.debug("reset_image_position callback complete")

    def event_track_registration(self, *args):
        
        self.event_change_color()
        for actor_name, actor in self.mesh_actors.items():
            logger.debug(f"<Actor {actor_name}> RT: \n{actor.user_matrix}")
    
    def event_realign_meshes(self, *args, main_mesh=None, other_meshes=[]):
        
        objs = {'fix' : main_mesh,
                'move': other_meshes}
        
        transformation_matrix = self.mesh_actors[f"{objs['fix']}"].user_matrix
        
        for obj in objs['move']:
            self.mesh_actors[f"{obj}"].user_matrix = transformation_matrix
            self.pv_plotter.add_actor(self.mesh_actors[f"{obj}"], name=obj)
        
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
                transformed_points = utils.transform_vertices(transformation_matrix, self.mesh_polydata[f'{actor_name}'].points)
                colors = utils.color_mesh(transformed_points.T)
                self.mesh_polydata[f'{actor_name}'].point_data.set_scalars(colors)
                
                mesh = self.pv_plotter.add_mesh(self.mesh_polydata[f'{actor_name}'], rgb=True, name=actor_name)
                mesh.user_matrix = transformation_matrix
                
                actor, _ = self.pv_plotter.add_actor(mesh, name=actor_name)
                
                # Save the new actor to a container
                container[actor_name] = actor

            self.mesh_actors = container
            
            logger.debug("event_change_color callback complete")
        else:
            logger.error("reference not set")
    
    def plot(self):

        self.pv_plotter.enable_joystick_actor_style()

        # Register callbacks
        self.pv_plotter.add_key_event('c', self.event_reset_camera)
        self.pv_plotter.add_key_event('z', self.event_zoom_out)
        self.pv_plotter.add_key_event('d', self.event_reset_image_position)
        self.pv_plotter.add_key_event('t', self.event_track_registration)

        for main_mesh, mesh_data in self.binded_meshes.items():
            event_func = functools.partial(self.event_realign_meshes, main_mesh=main_mesh, other_meshes=mesh_data['meshes'])
            self.pv_plotter.add_key_event(mesh_data['key'], event_func)
        
        self.pv_plotter.add_key_event('k', self.event_gt_position)
        self.pv_plotter.add_key_event('l', self.event_change_gt_position)
        self.pv_plotter.add_key_event('v', self.event_change_color)
        
        # Set the camera initial parameters
        self.pv_plotter.camera = self.xyviewcamera.copy()
        
        if self.register:
            self.pv_plotter.add_axes()
            # add the camera orientation to move the camera
            _ = self.pv_plotter.add_camera_orientation_widget()
            # Actual presenting
            cpos = self.pv_plotter.show(title="vision6D", return_cpos=True)
        else:
            self.pv_plotter.disable()
            cpos = self.pv_plotter.show(title="vision6D", return_cpos=True)
            result = self.pv_plotter.last_image
            res_plot = Image.fromarray(result)
            res_plot.save("test/data/res_plot.png")
        
        # logger.debug(f"\nrt: \n{self.mesh_actors['ossicles'].user_matrix}")
        logger.debug(f"\ncpos: {cpos}")
        
    def render_scene(self, scene_path:pathlib.Path, scale_factor:Tuple[float], render_image:bool, render_one_object:str='', with_mask:bool=True):
        
        self.pv_render.enable_joystick_actor_style()
        background = pv.read(scene_path) #"test/data/black_background.jpg"
        background = background.scale(scale_factor, inplace=False)
        
        if render_image:
           self.pv_render.add_mesh(background, rgb=True, opacity=1, name="image")
        else:
            self.pv_render.set_background('white')
            # generate white image
            self.pv_render.add_mesh(background, rgb=True, opacity=0, name="image")
            # generate grey image
            # image = self.pv_render.add_mesh(background, rgb=True, opacity=0.5, name="image")
            
            # read the mesh file
            if len(render_one_object) != 0:
                mesh = self.pv_render.add_mesh(self.mesh_polydata[f"{render_one_object}"], rgb=True)
                mesh.user_matrix = self.transformation_matrix
            else:
                for _, mesh_data in self.mesh_polydata.items():
                    mesh = self.pv_render.add_mesh(mesh_data, rgb=True)
                    mesh.user_matrix = self.transformation_matrix
        
        self.pv_render.camera = self.xyviewcamera.copy()
        self.pv_render.disable()
        self.pv_render.show()
        
        return self.pv_render.last_image
