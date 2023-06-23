
from .singleton import Singleton

import pyvista as pv
import numpy as np
import PIL.Image
import pathlib

from .. import utils
from .plot_store import PlotStore

class ImageStore(metaclass=Singleton):

    def __init__(self):

        self.plot_store = PlotStore()
        self.reset()
        
    def reset(self):
        self.image_path = None
        self.image_actor = None
        self.image_opacity = 0.8
        self.h = None
        self.w = None

    def add_image(self, image_source):
        if isinstance(image_source, pathlib.WindowsPath) or isinstance(image_source, str):
            self.image_path = str(image_source)
            image_source = np.array(PIL.Image.open(image_source), dtype='uint8')
        if len(image_source.shape) == 2: image_source = image_source[..., None]

        if self.plot_store.mirror_x: image_source = image_source[:, ::-1, :]
        if self.plot_store.mirror_y: image_source = image_source[::-1, :, :]

        dim = image_source.shape
        self.h, self.w, channel = dim[0], dim[1], dim[2]

        self.render = utils.create_render(self.w, self.h)

        image = pv.UniformGrid(dimensions=(self.w, self.h, 1), spacing=[0.01, 0.01, 1], origin=(0.0, 0.0, 0.0))
        image.point_data["values"] = image_source.reshape((self.w * self.h, channel)) # order = 'C
        image = image.translate(-1 * np.array(image.center), inplace=False)

        # Then add it to the plotter
        image = self.plot_store.plotter.add_mesh(image, cmap='gray', opacity=self.image_opacity, name='image') if channel == 1 else self.plot_store.plotter.add_mesh(image, rgb=True, opacity=self.image_opacity, name='image')
        actor, _ = self.plot_store.plotter.add_actor(image, pickable=False, name='image')
        # Save actor for later
        self.image_actor = actor
        
        # get the image scalar
        image_data = utils.get_image_actor_scalars(self.image_actor)
        assert (image_data == image_source).all() or (image_data*255 == image_source).all(), "image_data and image_source should be equal"

    def update_image_opacity(self, delta):
        self.set_image_opacity(self.image_opacity + delta)

    def set_image_opacity(self, image_opacity: float):
        np.clip(self.image_opacity, 0, 1)
        self.image_opacity = image_opacity
        self.image_actor.GetProperty().opacity = image_opacity
        self.plot_store.plotter.add_actor(self.image_actor, pickable=False, name='image')

    def calibrate_image(self):
        original_image = np.array(PIL.Image.open(self.image_path), dtype='uint8')
        # make the the original image shape is [h, w, 3] to match with the rendered calibrated_image
        original_image = original_image[..., :3]
        if len(original_image.shape) == 2: original_image = original_image[..., None]
        if original_image.shape[-1] == 1: original_image = np.dstack((original_image, original_image, original_image))
        calibrated_image = np.array(self.render_image(self.plotter.camera.copy()), dtype='uint8')
        return original_image, calibrated_image

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

    def remove_actor(self):
        image_actor = self.image_actor
        self.image_actor = None
        self.image_path = None
        return image_actor