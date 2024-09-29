'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: bbox_container.py
@time: 2023-07-03 20:26
@desc: create container for bounding box related actions in application
'''
import vtk
import pathlib
import matplotlib
import numpy as np
import pyvista as pv
from typing import Dict
from ..components import BboxModel
from .singleton import Singleton

class BboxContainer(metaclass=Singleton):
    def __init__(self, plotter):
        self.plotter = plotter
        self.reference = None
        self.bboxes: Dict[str, BboxModel] = {}
        self.bbox_model = BboxModel()

    def reset(self, name):
        self.bboxes[name].clear_attributes()
    
    def mirror_bbox(self, direction):
        if direction == 'x': self.bbox_model.mirror_x = not self.bbox_model.mirror_x
        elif direction == 'y': self.bbox_model.mirror_y = not self.bbox_model.mirror_y
        self.add_bbox(self.bbox_model.path)
                
    def add_bbox(self, bbox_source, image_center, w, h):
        if isinstance(bbox_source, pathlib.Path) or isinstance(bbox_source, str):
            self.bbox_model.path = str(bbox_source)
            self.bbox_model.name = pathlib.Path(self.bbox_model.path).stem
            bbox_source = np.load(bbox_source)
        
        # find the center of the image
        bbox_center = np.array([w//2, h//2, 0])

        # x1, y1, x2, y2
        if bbox_source.shape == (4, ): points = np.array([[bbox_source[0], bbox_source[1], 0], 
                                                        [bbox_source[0], bbox_source[3], 0], 
                                                        [bbox_source[2], bbox_source[3], 0], 
                                                        [bbox_source[2], bbox_source[1], 0]])
        elif bbox_source.shape == (4, 3): points = bbox_source

        self.bbox_model.source_obj = bbox_source
        self.bbox_model.width = w
        self.bbox_model.height = h
        
        # Consider the mirror effect
        if self.bbox_model.mirror_x: points[:, 0] = w - points[:, 0]
        if self.bbox_model.mirror_y: points[:, 1] = h - points[:, 1]
        
        # Due to camera view change to right handed coordinate system
        points = points - bbox_center - image_center
        cells = np.array([[2, 0, 1], [2, 1, 2], [2, 2, 3], [2, 3, 0]]).ravel()
        self.bbox_model.pv_obj = pv.UnstructuredGrid(cells, np.full((4,), vtk.VTK_LINE, dtype=np.uint8), points.astype(np.float32))
        self.bbox_model.opacity=0.5 
        self.bbox_model.previous_opacity=0.5
        # Add bbox surface object to the plot
        bbox_mesh = self.plotter.add_mesh(self.bbox_model.pv_obj, color=self.bbox_model.color, opacity=self.bbox_model.opacity, line_width=2)
        actor, _ = self.plotter.add_actor(bbox_mesh, pickable=True, name=self.bbox_model.name)
        self.bbox_model.actor = actor
        self.bboxes[self.bbox_model.name] = self.bbox_model
        self.reference = self.bbox_model.name
        return self.bbox_model

    def set_bbox_color(self, color, name):
        self.bboxes[name].actor.GetMapper().SetScalarVisibility(0)
        self.bboxes[name].actor.GetProperty().SetColor(matplotlib.colors.to_rgb(color))

    def set_bbox_opacity(self, name:str, opacity: float):
        self.bboxes[name].previous_opacity = self.bboxes[name].opacity
        self.bboxes[name].opacity = opacity
        self.bboxes[name].actor.GetProperty().opacity = opacity
        
    def reset_bbox(self, name, image_center):
        if self.bboxes[name].path:
            self.bboxes[name].mirror_x = False
            self.bboxes[name].mirror_y = False
            _ = self.add_bbox(self.bboxes[name].path, image_center, self.bboxes[name].width, self.bboxes[name].height)

    def remove_bbox(self, name):
        del self.bboxes[name]