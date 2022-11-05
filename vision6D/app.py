import pathlib
import logging
import numpy as np

import pyvista as pv
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyvista.plotting.render_passes import RenderPasses

logger = logging.getLogger("vision6D")

class App:

    def __init__(self):

        # Show output
        self.pl = pv.Plotter()
        self.pr = pv.Plotter(lighting=None)
        self.actors = {}
        self.actor_attrs = {}

        # "xy" camera view
        self.xyviewcamera = pv.Camera()
        self.xyviewcamera.position = (9.6, 5.4, 40)
        self.xyviewcamera.focal_point = (9.6, 5.4, 0)
        self.xyviewcamera.up = (0.0, 1.0, 0.0)
        self.xyviewcamera.zoom = 1.0

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
        actor = self.pl.add_mesh(image, rgb=True, opacity=0.35)
        actor, actor_attr = self.pl.add_actor(actor, name="image")

        # Save actor for later
        self.actors["image"] = actor
        self.actors["image-origin"] = actor.copy()
        self.actor_attrs['image'] = actor_attr

    def degree2matrix(self, r: list, t: list):
        rot = R.from_euler("xyz", r, degrees=True)
        rot = rot.as_matrix()

        trans = np.array(t).reshape((-1, 1))
        matrix = np.vstack((np.hstack((rot, trans)), np.array([0, 0, 0, 1])))

        return matrix

    def load_mesh(self, mesh_path:pathlib.Path, name:str, rgb: bool = False):

        # Check if the file exists
        if not mesh_path.exists():
            logger.error(f"{mesh_path} does not exists")
            return None

        # Create image mesh and add it to the plotter
        mesh = pv.read(mesh_path)

        actor = self.pl.add_mesh(mesh, rgb=rgb)
        
        # actor.orientation = (0, 0, 0)
        # actor.position = (0, 0, 0)
        actor.orientation = (23.294214240721413, 40.41958189080352, 179.0723301417804)
        actor.position = (4.103567000349267, 33.911323263117126, 12.771560037870557)

        actor, actor_attr = self.pl.add_actor(actor, name=name)

        # Save actor for later
        self.actors[name] = actor
        self.actor_attrs[name] = actor_attr
        
        logger.info(f"\nossicles_orientation: {self.actors['ossicles'].orientation}")
        logger.info(f"\nossicles_orientation: {self.actors['ossicles'].position}")

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
            actor.orientation = (23.294214240721413, 40.41958189080352, 179.0723301417804)
            actor.position = (4.103567000349267, 33.911323263117126, 12.771560037870557)
            
        logger.debug("event_gt_position callback complete")

    def plot(self):

        self.pl.add_axes()
        self.pl.enable_joystick_actor_style()

        # Register callbacks
        self.pl.add_key_event('c', self.event_reset_camera)
        self.pl.add_key_event('d', self.event_reset_image_position)
        self.pl.add_key_event('t', self.event_track_registration)
        self.pl.add_key_event('g', self.event_realign_facial_nerve_chorda)
        self.pl.add_key_event('h', self.event_realign_facial_nerve_ossicles)
        self.pl.add_key_event('j', self.event_realign_chorda_ossicles)
        self.pl.add_key_event('k', self.event_gt_position)
        
        # add the camera orientation to move the camera
        _ = self.pl.add_camera_orientation_widget()
        
        # Set the camera initial parameters
        self.pl.camera = self.xyviewcamera.copy()
        
        # Actual presenting
        cpos = self.pl.show(title="vision6D", return_cpos=True)
        logger.debug(f"\ncpos: {cpos}")
        # self.pl.renderer

    def render(self, mesh_path, rgb):
    
        self.pr.enable_joystick_actor_style()
        self.pr.set_background('black')
        # self.pr.add_axes()
        
        mesh = pv.read(mesh_path)
        mesh = self.pr.add_mesh(mesh, rgb=rgb)
        mesh.orientation = (23.294214240721413, 40.41958189080352, 179.0723301417804)
        mesh.position = (4.103567000349267, 33.911323263117126, 12.771560037870557)
    
        self.pr.camera = self.xyviewcamera.copy()
        
        self.pr.disable()
        self.pr.show()
        print("hhh")