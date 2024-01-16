'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: image_store.py
@time: 2023-07-03 20:23
@desc: create store for image related base functions
'''
import copy
import math
import numpy as np
import pathlib
import pyvista as pv
import PIL.Image

from . import Singleton
from ..tools import utils
from ..path import PLOT_SIZE

class ImageStore(metaclass=Singleton):
    def __init__(self, plotter):
        self.plotter = plotter
        self.mirror_x = False
        self.mirror_y = False

        # camera related parameters, do not reset the camera!
        self.camera = pv.Camera()
        self.fx = 10000
        self.fy = 10000
        self.cx = PLOT_SIZE[0] // 2
        self.cy = PLOT_SIZE[1] // 2
        self.cam_viewup = (0, 0, 0)

        self.reset()
        
    def reset(self):
        # image related parameters
        self.image_path = None
        self.image_source = None
        self.image_pv = None
        self.image_actor = None
        self.image_opacity = 0.8
        self.width = None
        self.height = None

    #^ Camera related
    def reset_camera(self):
        self.plotter.camera = self.camera.copy()

    def zoom_in(self):
        self.plotter.camera.zoom(2)

    def zoom_out(self):
        self.plotter.camera.zoom(0.5)

    def set_camera_extrinsics(self):
        self.camera.SetPosition((0, 0, -1e-8)) # Set the camera position at the origin of the world coordinate system
        self.camera.SetFocalPoint((*self.camera.GetWindowCenter(),0)) # Get the camera window center
        self.camera.SetViewUp(self.cam_viewup)
    
    def set_camera_intrinsics(self):
        # Set camera intrinsic attribute
        self.camera_intrinsics = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
                
        # convert the principal point to window center (normalized coordinate system) and set it
        wcx = -2*(self.cx - float(PLOT_SIZE[0])/2) / PLOT_SIZE[0]
        wcy =  2*(self.cy - float(PLOT_SIZE[1])/2) / PLOT_SIZE[1]
        self.camera.SetWindowCenter(wcx, wcy) # (0,0)
        
        # Setting the view angle in degrees
        view_angle = (180 / math.pi) * (2.0 * math.atan2(PLOT_SIZE[1]/2.0, self.fx)) # or view_angle = np.degrees(2.0 * math.atan2(height/2.0, f)) # focal_length = (height / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
        self.camera.SetViewAngle(view_angle) # view angle should be in degrees

    def set_camera_props(self):
        self.set_camera_intrinsics()
        self.set_camera_extrinsics()
        self.reset_camera()

    #^ Set the plot size and update the camera intrinsics
    def set_plot_size(self, width, height):
        global PLOT_SIZE  # Declare that we are referring to the global variable
        PLOT_SIZE = (width, height)  # This will update the global variable
        self.cx = PLOT_SIZE[0] // 2
        self.cy = PLOT_SIZE[1] // 2
        self.set_camera_intrinsics()
    
    #^ Image related
    def add_image(self, image_source, object_distance):
        if isinstance(image_source, pathlib.Path) or isinstance(image_source, str):
            self.image_path = str(image_source)
            image_source = np.array(PIL.Image.open(image_source), dtype='uint8')
            
        """
        1. Image Origin: Traditional image representations consider the top-left corner as the origin (0, 0). 
        The first coordinate goes from top to bottom (rows or height) and the second from left to right (columns or width). 
        Thus, the y-axis effectively points downward in traditional 2D image coordinates.
        2. 3D Coordinate Systems: In a right-handed 3D coordinate system, the y-axis typically points upwards. 
        So, when we visualize a 2D image in a 3D space without adjustments, the image will appear flipped along the horizontal axis.
        """
        image_source = np.fliplr(np.flipud(image_source))
        
        #^ save the image_source for mirroring image in the video
        self.image_source = copy.deepcopy(image_source)

        if len(image_source.shape) == 2: image_source = image_source[..., None]
        dim = image_source.shape
        # set up the canvas size based on the image size
        self.set_plot_size(dim[1], dim[0]) # make sure is (width, height)
        self.reset_camera()

        self.height, self.width, channel = dim[0], dim[1], dim[2]

        # Consider the mirror effect with the preprocessed image_source: image_source = np.fliplr(np.flipud(image_source))
        if self.mirror_x: image_source = image_source[::-1, :, :]
        if self.mirror_y: image_source = image_source[:, ::-1, :]

        self.render = utils.create_render(self.width, self.height)
        
        # create the image pyvista object
        self.image_pv = pv.ImageData(dimensions=(self.width, self.height, 1), spacing=[0.01, 0.01, 1], origin=(0.0, 0.0, 0.0))
        self.image_pv.point_data["values"] = image_source.reshape((self.width * self.height, channel)) # order = 'C
        self.image_pv = self.image_pv.translate(-1 * np.array(self.image_pv.center), inplace=False) # center the image at (0, 0)
        self.image_pv.translate(np.array([0, 0, object_distance]), inplace=True) # move the image to the camera distance
        
        return self.image_pv, image_source, channel
    
    def update_opacity(self, delta):
        self.image_opacity += delta
        self.image_opacity = np.clip(self.image_opacity, 0, 1)
        self.image_actor.GetProperty().opacity = self.image_opacity
        
    def render_image(self, camera):
        self.render.clear()
        render_actor = self.image_actor.copy(deep=True)
        render_actor.GetProperty().opacity = 1
        self.render.add_actor(render_actor, pickable=False)
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        image = self.render.last_image
        return image