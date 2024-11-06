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

    def reset(self, name):
        self.bboxes[name].clear_attributes()
    
    # def mirror_bbox(self, name, direction):
    #     bbox_model = self.bboxes[name]
    #     if direction == 'x': bbox_model.mirror_x = not bbox_model.mirror_x
    #     elif direction == 'y': bbox_model.mirror_y = not bbox_model.mirror_y
    #     self.add_bbox(bbox_model.path)
                
    def add_bbox(self, bbox_source, image_center, w, h):
        # Create a new BboxModel instance
        bbox_model = BboxModel()

        if isinstance(bbox_source, pathlib.Path) or isinstance(bbox_source, str):
            bbox_model.path = str(bbox_source)
            bbox_model.name = pathlib.Path(bbox_model.path).stem
            bbox_source = np.load(bbox_source)
        
        # find the center of the image
        bbox_center = np.array([w//2, h//2, 0])

        # x1, y1, x2, y2
        if bbox_source.shape == (4, ): points = np.array([[bbox_source[0], bbox_source[1], 0], 
                                                        [bbox_source[0], bbox_source[3], 0], 
                                                        [bbox_source[2], bbox_source[3], 0], 
                                                        [bbox_source[2], bbox_source[1], 0]])
        elif bbox_source.shape == (4, 3): points = bbox_source

        bbox_model.source_obj = bbox_source
        bbox_model.width = w
        bbox_model.height = h
        
        # Consider the mirror effect
        # if bbox_model.mirror_x: points[:, 0] = w - points[:, 0]
        # if bbox_model.mirror_y: points[:, 1] = h - points[:, 1]
        
        # Due to camera view change to right handed coordinate system
        points = points - bbox_center - image_center
        cells = np.array([[2, 0, 1], [2, 1, 2], [2, 2, 3], [2, 3, 0]]).ravel()
        bbox_model.pv_obj = pv.UnstructuredGrid(cells, np.full((4,), vtk.VTK_LINE, dtype=np.uint8), points.astype(np.float32))
        bbox_model.opacity=0.5 
        bbox_model.previous_opacity=0.5
        # Add bbox surface object to the plot
        bbox_mesh = self.plotter.add_mesh(bbox_model.pv_obj, color=bbox_model.color, opacity=bbox_model.opacity, line_width=2)
        actor, _ = self.plotter.add_actor(bbox_mesh, pickable=True, name=bbox_model.name)
        bbox_model.actor = actor
        self.bboxes[bbox_model.name] = bbox_model
        return bbox_model

    def set_bbox_color(self, color, name):
        bbox_model = self.bboxes[name]
        bbox_model.actor.GetMapper().SetScalarVisibility(0)
        bbox_model.actor.GetProperty().SetColor(matplotlib.colors.to_rgb(color))

    def set_bbox_opacity(self, name:str, opacity: float):
        bbox_model = self.bboxes[name]
        bbox_model.previous_opacity = bbox_model.opacity
        bbox_model.opacity = opacity
        bbox_model.actor.GetProperty().opacity = opacity
        
    def reset_bbox(self, name, image_center):
        bbox_model = self.bboxes[name]
        if bbox_model.path:
            _ = self.add_bbox(bbox_model.path, image_center, bbox_model.width, bbox_model.height)