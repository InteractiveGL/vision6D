import numpy as np
import pathlib
import pyvista as pv
import PIL.Image


from . import Singleton
from .. import utils

# contains mesh objects

class ImageStore(metaclass=Singleton):
    def __init__(self):
        self.reset()

    def reset(self):
        self.image_path = None
        self.image_actor = None
        self.image_opacity = 0.8

    def update_opacity(self, image_opacity: float):
        np.clip(image_opacity, 0, 1)
        self.image_opacity = image_opacity
        self.image_actor.GetProperty().opacity = image_opacity

    def add_image(self, image_source, mirror_x, mirror_y):
        if isinstance(image_source, pathlib.WindowsPath) or isinstance(image_source, str):
            self.image_path = str(image_source)
            image_source = np.array(PIL.Image.open(image_source), dtype='uint8')
        if len(image_source.shape) == 2: image_source = image_source[..., None]

        if mirror_x: image_source = image_source[:, ::-1, :]
        if mirror_y: image_source = image_source[::-1, :, :]

        dim = image_source.shape
        h, w, channel = dim[0], dim[1], dim[2]

        self.render = utils.create_render(w, h)

        image = pv.UniformGrid(dimensions=(w, h, 1), spacing=[0.01, 0.01, 1], origin=(0.0, 0.0, 0.0))
        image.point_data["values"] = image_source.reshape((w * h, channel)) # order = 'C
        image = image.translate(-1 * np.array(image.center), inplace=False)

        return image, image_source, channel
    
    def set_opacity(self, image_opacity):
        np.clip(image_opacity, 0, 1), "image opacity should range from 0 to 1!"
        self.image_opacity = image_opacity
        self.image_actor.GetProperty().opacity = image_opacity
    
    def update_opacity(self, delta):
        image_opacity += delta
        self.set_opacity(image_opacity)
        
    def render(self, camera):
        self.render.clear()
        render_actor = self.image_actor.copy(deep=True)
        render_actor.GetProperty().opacity = 1
        self.render.add_actor(render_actor, pickable=False)
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        image = self.render.last_image
        return image