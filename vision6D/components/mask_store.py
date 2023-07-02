import pathlib

import cv2
import PIL.Image
import numpy as np
import pyvista as pv

from . import Singleton
from .. import utils

# contains mesh objects

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
        points2d = contours[0].squeeze()
        
        # Mirror points
        h, w = mask_source.shape[0], mask_source.shape[1]
        
        self.render = utils.create_render(w, h)
        
        if self.mirror_x: points2d[:, 0] = w - points2d[:, 0]
        if self.mirror_y: points2d[:, 1] = h - points2d[:, 1]

        bottom_point = points2d[np.argmax(points2d[:, 1])]

        mask_center = (mask_source.shape[1] // 2, mask_source.shape[0] // 2)

        self.mask_offset = np.hstack(((bottom_point - mask_center)*0.01, 0))

        # Pad points a z dimension
        points = np.hstack((points2d*0.01, np.zeros(points2d.shape[0]).reshape((-1, 1))))
        # Find the bottom point on mask
        self.mask_bottom_point = points[np.argmax(points[:, 1])]

        # Create the mesh surface object
        cells = np.hstack([[points.shape[0]], np.arange(points.shape[0]), 0])
        mask_surface = pv.PolyData(points, cells).triangulate()
        mask_surface = mask_surface.translate(-self.mask_bottom_point+self.mask_offset, inplace=False)

        return mask_surface, points

    def update_opacity(self, delta):
        self.mask_opacity += delta
        self.mask_opacity = np.clip(self.mask_opacity, 0, 1)
        self.mask_actor.GetProperty().opacity = self.mask_opacity

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