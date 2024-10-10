'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: mask_container.py
@time: 2023-07-03 20:26
@desc: create container for mask related actions in application
'''
import cv2
import numpy as np
import pathlib
import pyvista as pv
from PIL import Image
import matplotlib.colors
from typing import Dict
from ..components import MaskModel
from ..tools import utils
from .singleton import Singleton

class MaskContainer(metaclass=Singleton):
    def __init__(self, plotter):
        self.plotter = plotter
        self.reference = None
        self.masks: Dict[str, MaskModel] = {}
        self.mask_model = MaskModel()

    def reset(self, name):
        self.masks[name].clear_attributes()

    def mirror_mask(self, direction):
        if direction == 'x': self.mask_model.mirror_x = not self.mask_model.mirror_x
        elif direction == 'y': self.mask_model.mirror_y = not self.mask_model.mirror_y
        self.add_mask(self.mask_model.path)

    def add_mask(self, mask_source, image_center, w, h):
        if isinstance(mask_source, pathlib.Path) or isinstance(mask_source, str):
            self.mask_model.path = str(mask_source)

        if pathlib.Path(self.mask_model.path).suffix == '.npy':
            points = np.load(self.mask_model.path).squeeze()
        else:
            mask_source = np.array(Image.open(mask_source), dtype='uint8')
            h, w = mask_source.shape[0], mask_source.shape[1]
            # if len(mask_source.shape) == 2: mask_source = mask_source[..., None]
            if mask_source.shape[-1] == 3: mask_source = cv2.cvtColor(mask_source, cv2.COLOR_RGB2GRAY)
            contours, _ = cv2.findContours(mask_source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points = contours[0].squeeze()
            self.mask_model.source_obj = mask_source

        self.mask_model.name = pathlib.Path(self.mask_model.path).stem
        while self.mask_model.name in self.masks: self.mask_model.name += "_copy"
        points = np.hstack((points, np.zeros(points.shape[0]).reshape((-1, 1))))

        self.mask_model.width = w
        self.mask_model.height = h
        
        # Mirror points
        mask_center = np.array([self.mask_model.width//2, self.mask_model.height//2, 0])
        self.render = utils.create_render(self.mask_model.width, self.mask_model.height)
        
        # Consider the mirror effect
        if self.mask_model.mirror_x: points[:, 0] = self.mask_model.width - points[:, 0]
        if self.mask_model.mirror_y: points[:, 1] = self.mask_model.height - points[:, 1]

        # Create the mesh surface object
        cells = np.hstack([[points.shape[0]], np.arange(points.shape[0]), 0])
        points = points - mask_center - image_center
        self.mask_model.mask_center = mask_center
        self.mask_model.image_center = image_center
        self.mask_model.pv_obj = pv.PolyData(points, cells).triangulate()
        self.mask_model.opacity = 0.5
        self.mask_model.previous_opacity = 0.5
        self.mask_model.color = 'white'
    
        mask_mesh = self.plotter.add_mesh(self.mask_model.pv_obj, color=self.mask_model.color, style='surface', opacity=self.mask_model.opacity)
        actor, _ = self.plotter.add_actor(mask_mesh, pickable=True, name='mask')
        self.mask_model.actor = actor

        self.masks[self.mask_model.name] = self.mask_model
        self.reference = self.mask_model.name
        return self.mask_model
    
    def load_mask(self):
        pass
    
    def reset_mask(self):
        self.masks[self.reference].mirror_x = False
        self.masks[self.reference].mirror_y = False
        _ = self.add_mask(mask_source=self.masks[self.reference].path, image_center=self.masks[self.reference].image_center, w = self.masks[self.reference].width, h = self.masks[self.reference].height)

    def set_mask_opacity(self, opacity: float):
        self.mask_model.previous_opacity = self.mask_model.opacity
        self.mask_model.opacity = opacity
        self.mask_model.actor.GetProperty().opacity = opacity

    def set_mask_color(self, name, color):
        self.masks[name].actor.GetMapper().SetScalarVisibility(0)
        self.masks[name].actor.GetProperty().SetColor(matplotlib.colors.to_rgb(color))
    
    def update_mask(self):
        tranformed_points = utils.get_mask_actor_points(self.mask_model.actor)
        cells = np.hstack([[tranformed_points.shape[0]], np.arange(tranformed_points.shape[0]), 0])
        mask_surface = pv.PolyData(tranformed_points, cells).triangulate()
        return mask_surface
    
    def remove_mask(self, name):
        del self.masks[name]

    def render_mask(self, camera):
        render = utils.create_render(self.mask_model.width, self.mask_model.height)
        render.clear()
        render_actor = self.mask_model.actor.copy(deep=True)
        render_actor.GetProperty().opacity = 1
        render.add_actor(render_actor, pickable=False)
        render.camera = camera
        render.disable()
        render.show(auto_close=False)
        image = render.last_image
        return image