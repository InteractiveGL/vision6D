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
        self.color = "white"
        self.color_button = None
        self.mask_path = None
        self.mask_pv = None
        self.mask_actor = None
        self.mask_opacity = 0.5
        self.previous_opacity = 0.5
        self.opacity_spinbox = None
    
    def add_mask(self, mask_source, image_center, size):
        w, h = size[0], size[1]

        if isinstance(mask_source, pathlib.Path) or isinstance(mask_source, str):
            self.mask_path = str(mask_source)

        if pathlib.Path(self.mask_path).suffix == '.npy':
            points = np.load(self.mask_path).squeeze()
        else:
            mask_source = np.array(PIL.Image.open(mask_source), dtype='uint8')
            h, w = mask_source.shape[0], mask_source.shape[1]
            # if len(mask_source.shape) == 2: mask_source = mask_source[..., None]
            if mask_source.shape[-1] == 3: mask_source = cv2.cvtColor(mask_source, cv2.COLOR_RGB2GRAY)
            # Get the segmentation contour points
            contours, _ = cv2.findContours(mask_source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points = contours[0].squeeze()

        points = np.hstack((points, np.zeros(points.shape[0]).reshape((-1, 1))))
        
        # Mirror points
        mask_center = np.array([w//2, h//2, 0])
        self.render = utils.create_render(w, h)
        
        # Consider the mirror effect
        if self.mirror_x: points[:, 0] = w - points[:, 0]
        if self.mirror_y: points[:, 1] = h - points[:, 1]

        # Create the mesh surface object
        cells = np.hstack([[points.shape[0]], np.arange(points.shape[0]), 0])
        # Due to camera view change to (0, -1, 0): x->right, y->down, z->front
        points = points - mask_center - image_center # equivalent to self.mask_pv.translate(np.array([(-w//2-image_center[0]), (-h//2-image_center[1]), object_distance]), inplace=True)
        self.mask_pv = pv.PolyData(points, cells).triangulate()
        # self.mask_pv.translate(np.array([(-w/2), (-h/2), object_distance]), inplace=True)
        # self.mask_pv.translate(-image_center, inplace=True)
        return self.mask_pv

    def update_mask(self):
        tranformed_points = utils.get_mask_actor_points(self.mask_actor)
        cells = np.hstack([[tranformed_points.shape[0]], np.arange(tranformed_points.shape[0]), 0])
        mask_surface = pv.PolyData(tranformed_points, cells).triangulate()
        return mask_surface

    def render_mask(self, camera):
        render = utils.create_render(self.w, self.h)
        render.clear()
        render_actor = self.mask_actor.copy(deep=True)
        render_actor.GetProperty().opacity = 1
        render.add_actor(render_actor, pickable=False)
        render.camera = camera
        render.disable()
        render.show(auto_close=False)
        image = render.last_image
        return image