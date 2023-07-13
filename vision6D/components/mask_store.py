'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: mask_store.py
@time: 2023-07-03 20:24
@desc: create store for mask related functions
'''

import pathlib

import cv2
import PIL.Image
import numpy as np
import pyvista as pv

from . import Singleton
from ..tools import utils

class MaskStore(metaclass=Singleton):
    def __init__(self):
        self.reset()
        self.mirror_x = False
        self.mirror_y = False

    def reset(self):
        self.mask_path = None
        self.mask_actor = None
        self.mask_opacity = 0.3
    
    def add_mask(self, mask_source):
        if isinstance(mask_source, pathlib.WindowsPath) or isinstance(mask_source, str):
            self.mask_path = str(mask_source)
            mask_source = np.array(PIL.Image.open(mask_source), dtype='uint8')

        # if len(mask_source.shape) == 2: mask_source = mask_source[..., None]
        if mask_source.shape[-1] == 3: mask_source = cv2.cvtColor(mask_source, cv2.COLOR_RGB2GRAY)

        # Get the segmentation contour points
        contours, _ = cv2.findContours(mask_source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = contours[0].squeeze()
        points = points * 0.01
        points = np.hstack((points, np.zeros(points.shape[0]).reshape((-1, 1))))
        
        # Mirror points
        h, w = mask_source.shape[0], mask_source.shape[1]
        
        self.render = utils.create_render(w, h)
        
        if self.mirror_x: points[:, 0] = w*0.01 - points[:, 0]
        if self.mirror_y: points[:, 1] = h*0.01 - points[:, 1]

        self.mask_bottom_point = points[np.argmax(points[:, 1])]
        mask_center = np.array([mask_source.shape[1] // 2, mask_source.shape[0] // 2, 0]) * 0.01
        self.mask_offset = self.mask_bottom_point - mask_center

        # Create the mesh surface object
        cells = np.hstack([[points.shape[0]], np.arange(points.shape[0]), 0])
        mask_surface = pv.PolyData(points, cells).triangulate()
        mask_surface = mask_surface.translate(-self.mask_bottom_point+self.mask_offset, inplace=False)

        return mask_surface

    def update_opacity(self, delta):
        self.mask_opacity += delta
        self.mask_opacity = np.clip(self.mask_opacity, 0, 1)
        self.mask_actor.GetProperty().opacity = self.mask_opacity

    def update_mask(self):
        tranformed_points = utils.get_mask_actor_points(self.mask_actor)
        cells = np.hstack([[tranformed_points.shape[0]], np.arange(tranformed_points.shape[0]), 0])
        mask_surface = pv.PolyData(tranformed_points, cells).triangulate()
        return mask_surface

    def render_mask(self, camera):
        self.render.clear()
        render_actor = self.mask_actor.copy(deep=True)
        render_actor.GetProperty().opacity = 1
        self.render.add_actor(render_actor, pickable=False)
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        image = self.render.last_image
        return image