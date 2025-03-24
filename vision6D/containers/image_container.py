'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: image_container.py
@time: 2023-07-03 20:26
@desc: create container for image related actions in application
'''
import pathlib
import PIL.Image
import numpy as np
import pyvista as pv
from .singleton import Singleton
from typing import Dict
from ..tools import utils
from ..components import ImageModel

class ImageContainer(metaclass=Singleton):
    def __init__(self, plotter):
        self.plotter = plotter
        self.reference = None
        self.images: Dict[str, ImageModel] = {}

    def reset(self, name):
        self.images[name].clear_attributes()

    def add_image_attributes(self, image_source):
        # Create a new ImageModel instance
        image_model = ImageModel()
        image_model.path = str(image_source)
        name = pathlib.Path(image_model.path).stem
        while name + "_image" in self.images: name = name + "_copy"
        image_model.name = name + "_image"
        image_model.source_obj = np.array(PIL.Image.open(image_model.path), dtype='uint8')

        if len(image_model.source_obj.shape) == 2: image_model.source_obj = image_model.source_obj[..., None]
        dim = image_model.source_obj.shape
        image_model.height = dim[0]
        image_model.width = dim[1]
        image_model.channel = dim[2]

        self.images[image_model.name] = image_model
        return image_model

    def add_image_actor(self, image_model, fy, cx, cy):
        # Create the image pyvista object
        pv_obj = pv.ImageData(dimensions=(image_model.width, image_model.height, 1), spacing=[1, 1, 1], origin=(0.0, 0.0, 0.0))
        pv_obj.point_data["values"] = image_model.source_obj.reshape((image_model.width * image_model.height, image_model.channel))
        pv_obj = pv_obj.translate(-np.array(pv_obj.center), inplace=False) # Center the image at (0, 0)
        
        # Compute offsets
        image_model.cx_offset = cx - (image_model.width / 2.0)
        image_model.cy_offset = cy - (image_model.height / 2.0)

        # Move the image to the camera distance fy
        image_model.distance2camera = fy
        pv_obj = pv_obj.translate(np.array([-image_model.cx_offset, -image_model.cy_offset, image_model.distance2camera]), inplace=False)

        # Add the mesh to the plotter
        if image_model.channel == 1: image_actor = self.plotter.add_mesh(pv_obj, cmap='gray', opacity=image_model.opacity, name=image_model.name)
        else: image_actor = self.plotter.add_mesh(pv_obj, rgb=True, opacity=image_model.opacity, pickable=False, name=image_model.name)

        # Store the actor in the image_model
        image_model.actor = image_actor

        # Update the image_model in self.images
        self.images[image_model.name] = image_model

    def set_image_opacity(self, name: str, opacity: float):
        if name in self.images:
            image_model = self.images[name]
            image_model.previous_opacity = image_model.opacity
            image_model.opacity = opacity
            if hasattr(image_model, 'actor'):
                image_model.actor.GetProperty().opacity = opacity

    def render_image(self, camera):
        image_model = self.images[self.reference]
        render = utils.create_render(image_model.width, image_model.height)
        render.clear()
        # Ensure actor exists
        if hasattr(image_model, 'actor'):
            render_actor = image_model.actor.copy(deep=True)
            render_actor.GetProperty().opacity = 1
            render.add_actor(render_actor, pickable=False)
            render.camera = camera
            render.disable()
            render.show(auto_close=False)
            image = render.last_image
            return image
        else:
            print(f"No actor found for image '{self.reference}'.")
            return None
