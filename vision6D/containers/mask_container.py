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

    def reset(self, name):
        self.masks[name].clear_attributes()

    def mirror_mask(self, mask_model, direction):
        if direction == 'x': mask_model.mirror_x = not mask_model.mirror_x
        elif direction == 'y': mask_model.mirror_y = not mask_model.mirror_y
        self.add_mask(mask_model.path)

    def add_mask(self, mask_source, image_center, w, h):
        # Create a new MaskModel instance
        mask_model = MaskModel()

        if isinstance(mask_source, pathlib.Path) or isinstance(mask_source, str):
            mask_model.path = str(mask_source)

        if pathlib.Path(mask_model.path).suffix == '.npy':
            points = np.load(mask_model.path).squeeze()
        else:
            mask_source = np.array(Image.open(mask_source), dtype='uint8')
            h, w = mask_source.shape[0], mask_source.shape[1]
            # if len(mask_source.shape) == 2: mask_source = mask_source[..., None]
            if mask_source.shape[-1] == 3: mask_source = cv2.cvtColor(mask_source, cv2.COLOR_RGB2GRAY)
            contours, _ = cv2.findContours(mask_source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points = contours[0].squeeze()
            mask_model.source_obj = mask_source

        mask_model.name = pathlib.Path(mask_model.path).stem
        while mask_model.name in self.masks: mask_model.name += "_copy"
        points = np.hstack((points, np.zeros(points.shape[0]).reshape((-1, 1))))

        mask_model.width = w
        mask_model.height = h
        
        # Mirror points
        mask_center = np.array([mask_model.width//2, mask_model.height//2, 0])
        self.render = utils.create_render(mask_model.width, mask_model.height)
        
        # Consider the mirror effect
        if mask_model.mirror_x: points[:, 0] = mask_model.width - points[:, 0]
        if mask_model.mirror_y: points[:, 1] = mask_model.height - points[:, 1]

        # Create the mesh surface object
        cells = np.hstack([[points.shape[0]], np.arange(points.shape[0]), 0])
        points = points - mask_center - image_center
        mask_model.mask_center = mask_center
        mask_model.image_center = image_center
        mask_model.pv_obj = pv.PolyData(points, cells).triangulate()
        mask_model.opacity = 0.5
        mask_model.previous_opacity = 0.5
        mask_model.color = 'white'
    
        mask_mesh = self.plotter.add_mesh(mask_model.pv_obj, color=mask_model.color, style='surface', opacity=mask_model.opacity)
        actor, _ = self.plotter.add_actor(mask_mesh, pickable=True, name='mask')
        mask_model.actor = actor

        self.masks[mask_model.name] = mask_model
        self.reference = mask_model.name
        return mask_model
    
    def load_mask(self):
        pass
    
    def reset_mask(self):
        self.masks[self.reference].mirror_x = False
        self.masks[self.reference].mirror_y = False
        _ = self.add_mask(mask_source=self.masks[self.reference].path, image_center=self.masks[self.reference].image_center, w = self.masks[self.reference].width, h = self.masks[self.reference].height)

    def set_mask_opacity(self, name, opacity: float):
        mask_model = self.masks[name]
        mask_model.previous_opacity = mask_model.opacity
        mask_model.opacity = opacity
        mask_model.actor.GetProperty().opacity = opacity

    def set_mask_color(self, name, color):
        self.masks[name].actor.GetMapper().SetScalarVisibility(0)
        self.masks[name].actor.GetProperty().SetColor(matplotlib.colors.to_rgb(color))
    
    def update_mask(self, name):
        mask_model = self.masks[name]
        transformed_points = utils.get_mask_actor_points(mask_model.actor)
        cells = np.hstack([[transformed_points.shape[0]], np.arange(transformed_points.shape[0]), 0])
        mask_surface = pv.PolyData(transformed_points, cells).triangulate()
        return mask_surface

    def render_mask(self, name, camera):
        mask_model = self.masks[name]
        render = utils.create_render(mask_model.width, mask_model.height)
        render.clear()
        render_actor = mask_model.actor.copy(deep=True)
        render_actor.GetProperty().opacity = 1
        render.add_actor(render_actor, pickable=False)
        render.camera = camera
        render.disable()
        render.show(auto_close=False)
        image = render.last_image
        return image