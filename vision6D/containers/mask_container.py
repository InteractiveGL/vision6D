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
        self.colors = ["cyan", "magenta", "yellow", "lime", "dodgerblue", "white", "black", "wheat"]

    def reset(self, name):
        self.masks[name].clear_attributes()

    def add_mask(self, mask_source, fy, cx, cy):
        # Create a new MaskModel instance
        mask_model = MaskModel()

        if isinstance(mask_source, pathlib.Path) or isinstance(mask_source, str):
            mask_model.path = str(mask_source)

        mask_source = np.array(Image.open(mask_source), dtype='uint8')
        mask_model.height, mask_model.width = mask_source.shape[0], mask_source.shape[1]
        # if len(mask_source.shape) == 2: mask_source = mask_source[..., None]
        if mask_source.shape[-1] == 3: mask_source = cv2.cvtColor(mask_source, cv2.COLOR_RGB2GRAY)
        # Binarize the grayscale image
        # _, mask_source = cv2.threshold(mask_source, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask_source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = contours[0].squeeze()
        mask_model.source_obj = mask_source

        mask_model.name = pathlib.Path(mask_model.path).stem
        while mask_model.name + "_mask" in self.masks: mask_model.name += "_copy"
        mask_model.name = mask_model.name + "_mask"
        points = np.hstack((points, np.zeros(points.shape[0]).reshape((-1, 1))))

        # Create the mesh surface object
        cells = np.hstack([[points.shape[0]], np.arange(points.shape[0]), 0])
        mask_model.pv_obj = pv.PolyData(points, cells).triangulate()
        mask_model.pv_obj = mask_model.pv_obj.translate(np.array([-cx, -cy, fy]), inplace=False)
        mask_model.opacity = 0.5
        mask_model.previous_opacity = 0.5
        mask_model.color = self.colors[len(self.masks) % len(self.colors)]
    
        mask_mesh = self.plotter.add_mesh(mask_model.pv_obj, color=mask_model.color, style='surface', opacity=mask_model.opacity, pickable=True, name=mask_model.name)
        mask_model.actor = mask_mesh

        self.masks[mask_model.name] = mask_model
        self.reference = mask_model.name
        return mask_model

    def set_mask_opacity(self, name, opacity: float):
        mask_model = self.masks[name]
        mask_model.previous_opacity = mask_model.opacity
        mask_model.opacity = opacity
        mask_model.actor.GetProperty().opacity = opacity

    def set_mask_color(self, name, color):
        mask_model = self.masks[name]
        mask_model.actor.GetMapper().SetScalarVisibility(0)
        mask_model.actor.GetProperty().SetColor(matplotlib.colors.to_rgb(color))
    
    def update_mask(self, name):
        mask_model = self.masks[name]
        transformed_points = utils.get_mask_actor_points(mask_model.actor)
        cells = np.hstack([[transformed_points.shape[0]], np.arange(transformed_points.shape[0]), 0])
        mask_surface = pv.PolyData(transformed_points, cells).triangulate()
        mask_model = self.masks[self.reference]
        mask_model.pv_obj = mask_surface
        mask_model.color = mask_model.color
        mask_mesh = self.plotter.add_mesh(mask_model.pv_obj, color=mask_model.color, style='surface', opacity=mask_model.opacity, pickable=True, name=mask_model.name)
        mask_model.actor = mask_mesh
        self.masks[mask_model.name] = mask_model

    def render_mask(self, camera, cx, cy):
        mask_model = self.masks[self.reference]
        render = utils.create_render(mask_model.width, mask_model.height)
        render.clear()
        render_obj = mask_model.pv_obj.copy(deep=True)
        render_obj = render_obj.translate(np.array([cx-mask_model.width/2, cy-mask_model.height/2, 0]), inplace=False)
        render.add_mesh(render_obj, color=mask_model.color, style='surface', opacity=1, name=mask_model.name)
        render.camera = camera
        render.disable()
        render.show(auto_close=False)
        image = render.last_image
        return image