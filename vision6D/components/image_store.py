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

class ImageStore(metaclass=Singleton):
    def __init__(self, plotter):
        self.plotter = plotter
        self.camera = pv.Camera() # camera related parameters, do not reset the camera!
        self.mirror_x = False
        self.mirror_y = False
        self.reset()
        
    def reset(self):
        # image related parameters
        self.image_path = None
        self.image_source = None
        self.image_pv = None
        self.image_actor = None
        self.image_opacity = 0.9
        self.previous_opacity = 0.9
        self.opacity_spinbox = None
        self.width = None
        self.height = None
        # self.fx = None
        # self.fy = None
        # self.cx = None
        # self.cy = None
        # self.cam_viewup = None

        # for cochlear implant dataset
        # self.fx = 18466.768907841793
        # self.fy = 19172.02089833029
        # self.cx = 954.4324739015676
        # self.cy = 538.2131876789998
        # self.cam_viewup = (0, -1, 0)

        # linemod dataset camera parameters
        self.fx = 572.4114
        self.fy = 573.57043
        self.cx = 325.2611
        self.cy = 242.04899
        self.cam_viewup = (0, -1, 0)

        # handal dataset camera parameters
        # self.fx = 1589.958
        # self.fy = 1590.548
        # self.cx = 957.475
        # self.cy = 714.920
        # self.cam_viewup = (0, -1, 0)

        # hb dataset camera parameters
        # self.fx = 537.4799
        # self.fy = 536.1447
        # self.cx = 318.8965
        # self.cy = 238.3781
        # self.cam_viewup = (0, -1, 0)

    #^ Camera related
    def reset_camera(self):
        self.plotter.camera = self.camera.copy()

    def zoom_in(self):
        self.plotter.camera.zoom(2)

    def zoom_out(self):
        self.plotter.camera.zoom(0.5)

    def set_camera_extrinsics(self):
        self.camera.SetPosition((0, 0, -1e-8)) # Set the camera position at the origin of the world coordinate system
        self.camera.SetFocalPoint((0, 0, 0)) # Get the camera window center
        self.camera.SetViewUp(self.cam_viewup)
    
    def set_camera_intrinsics(self):
        # Set camera intrinsic attribute
        self.camera_intrinsics = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        
        # Setting the view angle in degrees, self.height is the height of the image
        view_angle = (180 / math.pi) * (2.0 * math.atan2(self.height/2.0, self.fy)) # or view_angle = np.degrees(2.0 * math.atan2(height/2.0, f)) # focal_length = (height / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
        self.camera.SetViewAngle(view_angle) # view angle should be in degrees

    def set_camera_props(self):
        self.set_camera_intrinsics()
        self.set_camera_extrinsics()
        self.object_distance = self.fy # set the frame distance to the camera
        self.reset_camera()
    
    #^ Image related
    def add_image(self, image_source):
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
        #^ save the image_source for mirroring image in the video
        self.image_source = copy.deepcopy(image_source)

        if len(image_source.shape) == 2: image_source = image_source[..., None]
        dim = image_source.shape
        self.height, self.width, channel = dim[0], dim[1], dim[2]
        self.set_camera_props()

        # Consider the mirror effect with the preprocessed image_source: image_source = np.fliplr(np.flipud(image_source))
        if self.mirror_x: image_source = image_source[:, ::-1, :]
        if self.mirror_y: image_source = image_source[::-1, :, :]

        self.render = utils.create_render(self.width, self.height)
        
        # create the image pyvista object
        self.image_pv = pv.ImageData(dimensions=(self.width, self.height, 1), spacing=[1, 1, 1], origin=(0.0, 0.0, 0.0))
        self.image_pv.point_data["values"] = image_source.reshape((self.width * self.height, channel)) # order = 'C
        self.image_pv = self.image_pv.translate(-1 * np.array(self.image_pv.center), inplace=False) # center the image at (0, 0)
        """
        Do not need fx and fy (the focal lengths) for this calculation 
        because we are simply computing the offset of the principal point (cx, cy) in pixel space relative to 
        the center of the image and converting that to world space using the image spacing (1).
        Note that if image spacing is [1, 1], it means that each pixel in the x and y directions corresponds to a world unit of 1 in those directions.
        """
        self.cx_offset = (self.cx - (self.width / 2.0))
        self.cy_offset = (self.cy - (self.height / 2.0))
        # print(f"Image Origin: {self.cx_offset, self.cy_offset}")
        self.image_pv.translate(np.array([-self.cx_offset, -self.cy_offset, self.object_distance]), inplace=True) # move the image to the camera distance
        self.image_center = np.array([self.cx_offset, self.cy_offset, -self.object_distance])
        
        return self.image_pv, image_source, channel
        
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