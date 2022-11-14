import pathlib
import logging
import numpy as np
from PIL import Image

import pyvista as pv
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyvista.plotting.render_passes import RenderPasses

logger = logging.getLogger("vision6D")

class App:

    def __init__(self, register, image_path, scale_factor=[1,1,1]):
        
        self.register = register
        self.actors = {}
        self.actor_attrs = {}

        # "xy" camera view
        self.xyviewcamera = pv.Camera()
        self.xyviewcamera.focal_point = (9.6, 5.4, 0)
        self.xyviewcamera.up = (0.0, 1.0, 0.0)
        
        self.transformation_matrix = np.array(
            [[-0.66487539, -0.21262585, -0.71605235,  3.25029551],
            [ 0.08437209, -0.97387229,  0.21084143, 30.99098483],
            [-0.74217388,  0.07976845,  0.66544341, 14.47777792],
            [ 0.        ,  0.        ,  0.        ,  1.        ]])
                
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
        # self.plot()
        
    def load_image(self, image_path:pathlib.Path, scale_factor:list=[1,1,1]):

        # Check if the file exists
        if not image_path.exists():
            logger.error(f"{image_path} does not exists")
            return None

        # Create image mesh and add it to the plotter
        image = pv.read(image_path)

        # Scale down image
        image = image.scale(scale_factor, inplace=False)

        # Then add it to the plotter
        image = self.pl.add_mesh(image, rgb=True, opacity=0.35)
        actor, actor_attr = self.pl.add_actor(image, name="image")

        # Save actor for later
        self.actors["image"] = actor
        self.actors["image-origin"] = actor.copy()
        self.actor_attrs['image'] = actor_attr

    def load_mesh(self, mesh_path:pathlib.Path, name:str, rgb: bool = False):

        # Check if the file exists
        if not mesh_path.exists():
            logger.error(f"{mesh_path} does not exists")
            return None

        # Create image mesh and add it to the plotter
        mesh = pv.read(mesh_path)
        
        # mesh = mesh.rotate_x(6.835578651406617, inplace=False)
        # mesh = mesh.rotate_y(47.91692755829381, inplace=False)
        # mesh = mesh.rotate_z(172.76787223914218, inplace=False)
        # mesh = mesh.translate((2.5987030981091648, 31.039133701224685, 14.477777915423951), inplace=False)
        
        # # Apply transformation to the mesh vertices
        # vertices = mesh.points
        # ones = np.ones((vertices.shape[0], 1))
        # homogeneous_vertices = np.append(vertices, ones, axis=1)
        # transformed_vertices = (self.transformation_matrix @ homogeneous_vertices.T)[:3].T
        # mesh.points = transformed_vertices

        mesh = self.pl.add_mesh(mesh, rgb=rgb)
        
        mesh.user_matrix = self.transformation_matrix
        
        actor, actor_attr = self.pl.add_actor(mesh, name=name)

        # Save actor for later
        self.actors[name] = actor
        self.actor_attrs[name] = actor_attr
        
        logger.debug(f"\n{name} orientation: {self.actors[name].orientation}")
        logger.debug(f"\n{name} position: {self.actors[name].position}")
        
    def event_zoom_out(self, *args):
        self.pl.camera.zoom(0.5)
        logger.debug("event_zoom_out callback complete")

    def event_reset_camera(self, *args):
        self.pl.camera = self.xyviewcamera.copy()
        logger.debug("reset_camera_event callback complete")

    def event_reset_image_position(self, *args):
        self.actors["image"] = self.actors["image-origin"].copy() # have to use deepcopy to prevent change self.actors["image-origin"] content
        self.pl.add_actor(self.actors["image"], name="image")
        logger.debug("reset_image_position callback complete")

    def event_track_registration(self, *args):

        for actor_name, actor in self.actors.items():
            logger.debug(f"<Actor {actor_name}> RT: \n{actor.user_matrix}")
            
    def event_realign_facial_nerve_chorda(self, *args):
        
        objs = {'fix' : 'ossicles',
                'move': ['facial_nerve', 'chorda']}
        
        rt = self.actors[f"{objs['fix']}"].user_matrix
        
        # # obtain the original ossicles orientation and position
        # orientation = self.actors[f"{objs['fix']}"].orientation
        # position = self.actors[f"{objs['fix']}"].position
        
        for obj in objs['move']:
            # self.actors[f"{obj}"].orientation = orientation
            # self.actors[f"{obj}"].position = position
            self.actors[f"{obj}"].user_matrix = rt
        
        logger.debug("realign_facial_nerve_chorda callback complete")
        
    def event_realign_facial_nerve_ossicles(self, *args):
        
        objs = {'fix' : 'chorda',
                'move': ['facial_nerve', 'ossicles']}
        
        rt = self.actors[f"{objs['fix']}"].user_matrix
        
        for obj in objs['move']:
            self.actors[f"{obj}"].user_matrix = rt
        
        logger.debug("realign_facial_nerve_ossicles callback complete")
        
    def event_realign_chorda_ossicles(self, *args):
        
        objs = {'fix' : 'facial_nerve',
                'move': ['chorda', 'ossicles']}
        
        rt = self.actors[f"{objs['fix']}"].user_matrix
        
        for obj in objs['move']:
            self.actors[f"{obj}"].user_matrix = rt
        
        logger.debug("realign_chorda_ossicles callback complete")
        
    def event_gt_position(self, *args):
        
        for actor_name, actor in self.actors.items():
            if actor_name == "image":
                continue
            # actor.orientation = self.gt_orientation
            # actor.position = self.gt_position
            actor.user_matrix = self.transformation_matrix
            
        logger.debug("event_gt_position callback complete")
        
    def event_change_gt_position(self, *args):
        
        # self.gt_orientation = self.actors["ossicles"].orientation
        # self.gt_position = self.actors["ossicles"].position
        
        self.transformation_matrix = self.actors["ossicles"].user_matrix
        
        for actor_name, actor in self.actors.items():
            if actor_name == "image":
                continue
            # actor.orientation = self.gt_orientation
            # actor.position = self.gt_position
            actor.user_matrix = self.transformation_matrix
            
        logger.debug("event_change_gt_position callback complete")
        
    def plot(self):

        self.pl.enable_joystick_actor_style()

        # Register callbacks
        self.pl.add_key_event('c', self.event_reset_camera)
        self.pl.add_key_event('z', self.event_zoom_out)
        self.pl.add_key_event('d', self.event_reset_image_position)
        self.pl.add_key_event('t', self.event_track_registration)
        self.pl.add_key_event('g', self.event_realign_facial_nerve_chorda)
        self.pl.add_key_event('h', self.event_realign_facial_nerve_ossicles)
        self.pl.add_key_event('j', self.event_realign_chorda_ossicles)
        self.pl.add_key_event('k', self.event_gt_position)
        self.pl.add_key_event('l', self.event_change_gt_position)
        
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
        
        logger.debug(f"\nrt: \n{self.actors['ossicles'].user_matrix}")
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

    def render_ossicles(self, scale_factor, mesh_path, rgb):
    
        self.pr.enable_joystick_actor_style()
        self.pr.set_background('white')
        
        image = pv.read("black_background.jpg")
        image = image.scale(scale_factor, inplace=False)
        image = self.pr.add_mesh(image, rgb=rgb, opacity=0, show_scalar_bar=False)
        mesh = pv.read(mesh_path)
        mesh = self.pr.add_mesh(mesh, rgb=rgb)
        
        mesh.user_matrix = self.transformation_matrix
    
        self.pr.camera = self.xyviewcamera.copy()
        
        self.pr.disable()
        self.pr.show()
        result = self.pr.last_image
        res_render = Image.fromarray(result)
        res_render.save("res_render.png")
        print("hhh")
        
    def plot_render(self, scale_factor, render_path):
        self.plot()
        self.render_ossicles(scale_factor, render_path, rgb=True)