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

import numpy as np
import pathlib
import pyvista as pv
import PIL.Image


from . import Singleton
from ..tools import utils

class ImageStore(metaclass=Singleton):
    def __init__(self):
        self.reset()
        self.mirror_x = False
        self.mirror_y = False

    def reset(self):
        self.image_path = None
        self.image_source = None
        self.image_pv = None
        self.image_actor = None
        self.image_opacity = 0.8
        self.width = None
        self.height = None

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