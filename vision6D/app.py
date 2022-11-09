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

    def __init__(self, register):
        
        self.register = register
        self.actors = {}
        self.actor_attrs = {}

        # "xy" camera view
        self.xyviewcamera = pv.Camera()
        self.xyviewcamera.focal_point = (9.6, 5.4, 0)
        self.xyviewcamera.up = (0.0, 1.0, 0.0)
        
        self.gt_orientation = (35.57143478233399, 86.14563590414456, 163.22833630484539)
        self.gt_position = (-4.892817134622411, 36.41065351570653, -5.650059814317807)
        
        if self.register:
            self.pl = pv.Plotter(window_size=[1920, 1080])
            self.xyviewcamera.position = (9.6, 5.4, 40)
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
        
    # def degree2matrix(self, r: list, t: list):
    #     rot = R.from_euler("xyz", r, degrees=True)
    #     rot = rot.as_matrix()

    #     trans = np.array(t).reshape((-1, 1))
    #     matrix = np.vstack((np.hstack((rot, trans)), np.array([0, 0, 0, 1])))

    #     return matrix

    def load_mesh(self, mesh_path:pathlib.Path, name:str, rgb: bool = False):

        # Check if the file exists
        if not mesh_path.exists():
            logger.error(f"{mesh_path} does not exists")
            return None

        # Create image mesh and add it to the plotter
        mesh = pv.read(mesh_path)

        mesh = self.pl.add_mesh(mesh, rgb=rgb)
        
        # actor.orientation = (0, 0, 0)
        # actor.position = (0, 0, 0)
        mesh.orientation = self.gt_orientation
        mesh.position = self.gt_position

        actor, actor_attr = self.pl.add_actor(mesh, name=name)

        # Save actor for later
        self.actors[name] = actor
        self.actor_attrs[name] = actor_attr
        
        logger.debug(f"\n{name} orientation: {self.actors[name].orientation}")
        logger.debug(f"\n{name} position: {self.actors[name].position}")

    def event_reset_camera(self, *args):
        self.pl.camera = self.xyviewcamera.copy()
        logger.debug("reset_camera_event callback complete")

    def event_reset_image_position(self, *args):
        self.actors["image"] = self.actors["image-origin"].copy() # have to use deepcopy to prevent change self.actors["image-origin"] content
        self.pl.add_actor(self.actors["image"], name="image")
        logger.debug("reset_image_position callback complete")

    def event_track_registration(self, *args):

        for actor_name, actor in self.actors.items():
            logger.debug(f"<Actor {actor_name}> R: {actor.orientation}, T: {actor.position}")
            
    def event_realign_facial_nerve_chorda(self, *args):
        
        objs = {'fix' : 'ossicles',
                'move': ['facial_nerve', 'chorda']}
        
        # obtain the original ossicles orientation and position
        orientation = self.actors[f"{objs['fix']}"].orientation
        position = self.actors[f"{objs['fix']}"].position
        
        for obj in objs['move']:
            self.actors[f"{obj}"].orientation = orientation
            self.actors[f"{obj}"].position = position
        
        logger.debug("realign_facial_nerve_chorda callback complete")
        
    def event_realign_facial_nerve_ossicles(self, *args):
        
        objs = {'fix' : 'chorda',
                'move': ['facial_nerve', 'ossicles']}
        
        # obtain the original ossicles orientation and position
        orientation = self.actors[f"{objs['fix']}"].orientation
        position = self.actors[f"{objs['fix']}"].position
        
        for obj in objs['move']:
            self.actors[f"{obj}"].orientation = orientation
            self.actors[f"{obj}"].position = position
        
        logger.debug("realign_facial_nerve_ossicles callback complete")
        
    def event_realign_chorda_ossicles(self, *args):
        
        objs = {'fix' : 'facial_nerve',
                'move': ['chorda', 'ossicles']}
        
        # obtain the original ossicles orientation and position
        orientation = self.actors[f"{objs['fix']}"].orientation
        position = self.actors[f"{objs['fix']}"].position
        
        for obj in objs['move']:
            self.actors[f"{obj}"].orientation = orientation
            self.actors[f"{obj}"].position = position
        
        logger.debug("realign_chorda_ossicles callback complete")
        
    def event_gt_position(self, *args):
        
        for actor_name, actor in self.actors.items():
            if actor_name == "image":
                continue
            actor.orientation = self.gt_orientation
            actor.position = self.gt_position
            
        logger.debug("event_gt_position callback complete")
        
    def event_change_gt_position(self, *args):
        
        self.gt_orientation = self.actors["ossicles"].orientation
        self.gt_position = self.actors["ossicles"].position
        
        for actor_name, actor in self.actors.items():
            if actor_name == "image":
                continue
            actor.orientation = self.gt_orientation
            actor.position = self.gt_position
            
        logger.debug("event_change_gt_position callback complete")
        
    def plot(self):

        self.pl.enable_joystick_actor_style()

        # Register callbacks
        self.pl.add_key_event('c', self.event_reset_camera)
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
        
        mesh.orientation = self.gt_orientation
        mesh.position = self.gt_position
    
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