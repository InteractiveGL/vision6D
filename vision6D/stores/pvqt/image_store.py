
from ..singleton import Singleton
from ..paths_store import PathsStore

import pyvista as pv
import numpy as np
import PIL.Image
import pathlib

from ... import utils

class ImageStore(metaclass=Singleton):

    def __init__(self):
        self.paths_store = PathsStore()
        self.reset()
        
    def reset(self):
        self.image_actor = None
        self.image_opacity = 0.8

    def add_image(self, image_source):
        if isinstance(image_source, pathlib.WindowsPath) or isinstance(image_source, str):
            image_source = np.array(PIL.Image.open(image_source), dtype='uint8')
        if len(image_source.shape) == 2: image_source = image_source[..., None]

        if self.mirror_x: image_source = image_source[:, ::-1, :]
        if self.mirror_y: image_source = image_source[::-1, :, :]

        dim = image_source.shape
        h, w, channel = dim[0], dim[1], dim[2]

        # Create the render based on the image size
        self.render = pv.Plotter(window_size=[w, h], lighting=None, off_screen=True) 
        self.render.set_background('black'); assert self.render.background_color == "black", "render's background need to be black"

        image = pv.UniformGrid(dimensions=(w, h, 1), spacing=[0.01, 0.01, 1], origin=(0.0, 0.0, 0.0))
        image.point_data["values"] = image_source.reshape((w * h, channel)) # order = 'C
        image = image.translate(-1 * np.array(image.center), inplace=False)

        # Then add it to the plotter
        image = self.plotter.add_mesh(image, cmap='gray', opacity=self.image_opacity, name='image') if channel == 1 else self.plotter.add_mesh(image, rgb=True, opacity=self.image_opacity, name='image')
        actor, _ = self.plotter.add_actor(image, pickable=False, name='image')
        # Save actor for later
        self.image_actor = actor
        
        # get the image scalar
        image_data = utils.get_image_actor_scalars(self.image_actor)
        assert (image_data == image_source).all() or (image_data*255 == image_source).all(), "image_data and image_source should be equal"

    def update_image_opacity(self, delta):
        self.set_image_opacity(self.image_opacity + delta)

    def set_image_opacity(self, image_opacity: float):
        assert image_opacity>=0 and image_opacity<=1, "image opacity should range from 0 to 1!"
        self.image_opacity = image_opacity
        self.image_actor.GetProperty().opacity = image_opacity
        self.plot_store.plotter.add_actor(self.image_actor, pickable=False, name='image')

    def remove_actor(self):
        image_actor = self.image_actor
        self.image_actor = None
        self.paths_store.image_path = None
        return image_actor