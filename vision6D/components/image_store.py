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
        self.image_actor = None
        self.image_opacity = 0.8
        self.width = None
        self.height = None

    def add_image(self, image_source):
        if isinstance(image_source, pathlib.Path) or isinstance(image_source, str):
            self.image_path = str(image_source)
            image_source = np.array(PIL.Image.open(image_source), dtype='uint8')
        
        #^ save the image_source for mirroring image in the video
        self.image_source = copy.deepcopy(image_source)

        if len(image_source.shape) == 2: image_source = image_source[..., None]
        dim = image_source.shape
        self.height, self.width, channel = dim[0], dim[1], dim[2]

        if self.mirror_x: image_source = image_source[:, ::-1, :]
        if self.mirror_y: image_source = image_source[::-1, :, :]

        self.render = utils.create_render(self.width, self.height)
        image = self.update_image(image_source, channel)
        return image, image_source, channel
    
    def update_image(self, image_source, channel):
        image = pv.ImageData(dimensions=(self.width, self.height, 1), spacing=[0.01, 0.01, 1], origin=(0.0, 0.0, 0.0))
        image.point_data["values"] = image_source.reshape((self.width * self.height, channel)) # order = 'C
        image = image.translate(-1 * np.array(image.center), inplace=False)
        return image
        
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