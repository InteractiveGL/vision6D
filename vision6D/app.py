import pathlib
import logging
import numpy as np

import pyvista as pv
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger("vision6D")

class App:

    def __init__(self):

        # Show output
        self.pl = pv.Plotter()
        # self.pr = pv.Renderer()
        self.actors = {}
        self.actor_attrs = {}

        # "xy" camera view
        self.xyviewcamera = pv.Camera()
        self.xyviewcamera.position = (7.75202772140503, 3.917879838943482, 53.657579687507386)
        self.xyviewcamera.focal_point = (7.75202772140503, 3.917879838943482, 0.4370880126953125)
        self.xyviewcamera.up = (0.0, 1.0, 0.0)
        self.xyviewcamera.zoom = 1.0


    def load_image(self, image_path:pathlib.Path, scaling:list=[1,1,1]):

        # Check if the file exists
        if not image_path.exists():
            logger.error(f"{image_path} does not exists")
            return None

        # Create image mesh and add it to the plotter
        image = pv.read(image_path)

        # Scale down image
        image = image.scale(scaling, inplace=False)

        # Then add it to the plotter
        actor = self.pl.add_mesh(image, rgb=True, opacity=0.65)
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

        # r = [8.770189620706784, 171.674855383245, 49.08692066857242]
        # t = [11.83933198174675, 5.512584484779442, -4.1060903869260095]

        # transform_matrix = self.convert2transform_matrix(r, t)

        # mesh.rotate_x(8.770189620706784, inplace=True)
        # mesh.rotate_y(171.674855383245, inplace=True)
        # mesh.rotate_z(49.08692066857242, inplace=True)

        # mesh = mesh.transform(transform_matrix)

        actor = self.pl.add_mesh(mesh, rgb=rgb)

        # actor.orientation = (21.84230194065212, 44.86586576118945, 171.99906466676183)
        # actor.position = (4.192207212217855, 7.394309210439285, -0.8389455453076233)

        actor.orientation = (13.93752422036519, 49.44108090502569, 170.56188741343297)
        actor.position = (4.212772778897769, 31.817706821593006, 15.391170375030706)

        actor, actor_attr = self.pl.add_actor(actor, name=name)

        # Save actor for later
        self.actors[name] = actor
        self.actor_attrs[name] = actor_attr

    def reset_camera_event(self, *args):
        self.pl.camera = self.xyviewcamera.copy()
        logger.debug("reset_camera_event callback complete")

    def reset_image_position(self, *args):
        self.actors["image"] = self.actors["image-origin"].copy() # have to use deepcopy to prevent change self.actors["image-origin"] content
        self.pl.add_actor(self.actors["image"], name="image")
        logger.debug("reset_image_position callback complete")

    def track_registration(self, *args):

        for actor_name, actor in self.actors.items():
            logger.debug(f"<Actor {actor_name}> R: {actor.orientation}, T: {actor.position}")
            
    def realign_other_meshes(self, *args):
        orientation = self.actors['ossicles'].orientation
        position = self.actors['ossicles'].position
        
        # realign facial nerve
        self.actors['facial_nerve'].orientation = orientation
        self.actors['facial_nerve'].position = position
        
        # realign chorda
        self.actors['chorda'].orientation = orientation
        self.actors['chorda'].position = position
        
        logger.debug("realign_other_meshes callback complete")

    def plot(self):

        self.pl.add_axes()
        self.pl.enable_joystick_actor_style()

        # Register callbacks
        self.pl.add_key_event('c', self.reset_camera_event)
        self.pl.add_key_event('d', self.reset_image_position)
        self.pl.add_key_event('t', self.track_registration)
        self.pl.add_key_event('g', self.realign_other_meshes)

        # add the camera orientation to move the camera
        _ = self.pl.add_camera_orientation_widget()
        
        # remove scalar bar
        self.pl.remove_scalar_bar()

        # Actual presenting
        cpos = self.pl.show(cpos="xy", return_cpos=True)
        logger.debug(f"\ncpos: {cpos}")
        logger.debug(f"\nossicles_orientation: {self.actors['ossicles'].orientation}")
        logger.debug(f"\nossicles_orientation: {self.actors['ossicles'].position}")

    def render(self):
        ...
        
        # Create image mesh and add it to the plotter
        # mesh = pv.read(mesh_path)

        # # actor = self.pl.add_mesh(mesh, rgb=True)

        # # actor.orientation = (13.93752422036519, 49.441080905025686, 170.56188741343297)
        # # actor.position = (4.4715937628911275, 7.283363364481435, -0.7639411050175622)

        # actor, actor_attr = self.pr.add_actor(mesh, name=name)
        
        # self.pr.view_xy()

        # self.pl.add_axes()
        # self.pl.enable_joystick_actor_style()

        # # Register callbacks
        # self.pl.add_key_event('c', self.reset_camera_event)
        # self.pl.add_key_event('d', self.reset_image_position)
        # self.pl.add_key_event('t', self.track_registration)

        # # add the camera orientation to move the camera
        # _ = self.pl.add_camera_orientation_widget()

        # # Actual presenting
        # cpos = self.pl.show(cpos="xy", return_cpos=True)
        # logger.debug(f"\ncpos: {cpos}")
        # logger.debug(f"\nossicles_orientation: {self.actors['ossicles'].orientation}")
        # logger.debug(f"\nossicles_orientation: {self.actors['ossicles'].position}")