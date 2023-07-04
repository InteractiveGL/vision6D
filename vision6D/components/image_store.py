'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: image_store.py
@time: 2023-07-03 20:23
@desc: create store for image related base functions
'''

import numpy as np
import pathlib
import pyvista as pv
import PIL.Image


from . import Singleton
from ..tools import utils

# contains mesh objects

class ImageStore(metaclass=Singleton):
    def __init__(self):
        self.reset()
        self.mirror_x = False
        self.mirror_y = False

    def reset(self):
        self.image_path = None
        self.image_actor = None
        self.image_opacity = 0.8

    def add_image(self, image_source):
        if isinstance(image_source, pathlib.WindowsPath) or isinstance(image_source, str):
            self.image_path = str(image_source)
            image_source = np.array(PIL.Image.open(image_source), dtype='uint8')
        if len(image_source.shape) == 2: image_source = image_source[..., None]

        if self.mirror_x: image_source = image_source[:, ::-1, :]
        if self.mirror_y: image_source = image_source[::-1, :, :]

        dim = image_source.shape
        h, w, channel = dim[0], dim[1], dim[2]

        self.render = utils.create_render(w, h)

        image = pv.UniformGrid(dimensions=(w, h, 1), spacing=[0.01, 0.01, 1], origin=(0.0, 0.0, 0.0))
        image.point_data["values"] = image_source.reshape((w * h, channel)) # order = 'C
        image = image.translate(-1 * np.array(image.center), inplace=False)

        return image, image_source, channel
        
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