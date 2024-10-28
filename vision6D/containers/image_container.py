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

# # for cochlear implant dataset
# # self.fx = 18466.768907841793
# # self.fy = 19172.02089833029
# # self.cx = 954.4324739015676
# # self.cy = 538.2131876789998
# # self.cam_viewup = (0, -1, 0)

# # linemod dataset camera parameters
# self.fx = 572.4114
# self.fy = 573.57043
# self.cx = 325.2611
# self.cy = 242.04899
# self.cam_viewup = (0, -1, 0)

# # handal dataset camera parameters
# # self.fx = 1589.958
# # self.fy = 1590.548
# # self.cx = 957.475
# # self.cy = 714.920
# # self.cam_viewup = (0, -1, 0)

# # hb dataset camera parameters
# # self.fx = 537.4799
# # self.fy = 536.1447
# # self.cx = 318.8965
# # self.cy = 238.3781
# # self.cam_viewup = (0, -1, 0)
        
class ImageContainer(metaclass=Singleton):
    def __init__(self, plotter):
        self.plotter = plotter
        self.reference = None
        self.images: Dict[str, ImageModel] = {}

    def reset(self, name):
        self.images[name].clear_attributes()

    def add_image(self, image_source, fy, cx, cy):
        # Create a new ImageModel instance
        image_model = ImageModel()
        
        if isinstance(image_source, pathlib.Path) or isinstance(image_source, str):
            image_model.path = str(image_source)
            name = pathlib.Path(image_model.path).stem
            while name in self.images: name = name + "_copy"
            image_model.name = name
            image_model.source_obj = np.array(PIL.Image.open(image_model.path), dtype='uint8')
        else:
            image_model.source_obj = image_source

        if len(image_model.source_obj.shape) == 2: image_model.source_obj = image_model.source_obj[..., None]
        dim = image_model.source_obj.shape
        image_model.height = dim[0]
        image_model.width = dim[1]
        image_model.channel = dim[2]
        self.render = utils.create_render(image_model.width, image_model.height)

        if image_model.mirror_x: image_model.source_obj = image_model.source_obj[:, ::-1, :]
        if image_model.mirror_y: image_model.source_obj = image_model.source_obj[::-1, :, :]

        # create the image pyvista object
        image_model.pv_obj = pv.ImageData(dimensions=(image_model.width, image_model.height, 1), spacing=[1, 1, 1], origin=(0.0, 0.0, 0.0))
        image_model.pv_obj.point_data["values"] = image_model.source_obj.reshape((image_model.width * image_model.height, image_model.channel)) # order = 'C
        image_model.pv_obj = image_model.pv_obj.translate(-1 * np.array(image_model.pv_obj.center), inplace=False) # center the image at (0, 0)
        """
        Do not directly feed fx and fy (the focal lengths) for this calculation 
        because we are simply computing the offset of the principal point (cx, cy) in pixel space relative to 
        the center of the image and converting that to world space using the image spacing (1).
        Note that if image spacing is [1, 1], it means that each pixel in the x and y directions corresponds to a world unit of 1 in those directions.
        """
        image_model.cx_offset = (cx - (image_model.width / 2.0))
        image_model.cy_offset = (cy - (image_model.height / 2.0))
        print(f"Image Origin: {image_model.cx_offset, image_model.cy_offset}")
        # move the image to the camera distance
        image_model.pv_obj.translate(np.array([-image_model.cx_offset, -image_model.cy_offset, fy]), inplace=True)
        image_model.center = np.array([image_model.cx_offset, image_model.cy_offset, -fy])
        self.images[name] = image_model
        if image_model.channel == 1: image = self.plotter.add_mesh(image_model.pv_obj, cmap='gray', opacity=image_model.opacity, name=image_model.name)
        else: image = self.plotter.add_mesh(image_model.pv_obj, rgb=True, opacity=image_model.opacity, name=image_model.name)
        actor, _ = self.plotter.add_actor(image, pickable=False, name=image_model.name)
        image_model.actor = actor

        return image_model
                                          
    def set_image_opacity(self, name: str, opacity: float):
        self.images[name].previous_opacity = self.images[name].opacity
        self.images[name].opacity = opacity
        self.images[name].actor.GetProperty().opacity = opacity

    def remove_image(self, name):
        del self.images[name]

    def render_image(self, name, camera):
        self.render.clear()
        image_model = self.images[name]
        render_actor = image_model.actor.copy(deep=True)
        render_actor.GetProperty().opacity = 1
        self.render.add_actor(render_actor, pickable=False)
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        image = self.render.last_image
        return image