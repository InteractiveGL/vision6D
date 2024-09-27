'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: bbox_container.py
@time: 2023-07-03 20:26
@desc: create container for bounding box related actions in application
'''

import pathlib

import vtk
import matplotlib
import numpy as np
import pyvista as pv

from PyQt5 import QtWidgets

from ..tools import utils
from ..components import ImageStore
from ..components import BboxStore
from ..widgets import BboxWindow

class BboxContainer:
    def __init__(self, plotter):

        self.plotter = plotter
        self.image_store = ImageStore()
        self.bbox_store = BboxStore()
    
    def mirror_bbox(self, direction):
        if direction == 'x': self.bbox_store.mirror_x = not self.bbox_store.mirror_x
        elif direction == 'y': self.bbox_store.mirror_y = not self.bbox_store.mirror_y
        self.add_bbox(self.bbox_store.bbox_path)

    def load_bbox(self, bbox_data):
        # Add bbox surface object to the plot
        bbox_mesh = self.plotter.add_mesh(bbox_data.bbox_pv, color=bbox_data.color, opacity=bbox_data.opacity, line_width=2)
        actor, _ = self.plotter.add_actor(bbox_mesh, pickable=True, name=bbox_data.name)
        bbox_data.actor = actor
        return bbox_data
                
    def add_bbox(self, bbox_source):
        bbox_data = self.bbox_store.add_bbox(bbox_source, 
                                    self.image_store.images[self.image_store.reference].image_center, 
                                    self.image_store.images[self.image_store.reference].width, 
                                    self.image_store.images[self.image_store.reference].height)
        bbox_data = self.load_bbox(bbox_data)

        return bbox_data
    
    def draw_bbox(self):
        def handle_output_path_change(output_path):
            if output_path:
                self.bbox_store.bbox_path = output_path
                self.add_bbox(self.bbox_store.bbox_path)
        if self.image_store.images[self.image_store.reference].actor:
            image = utils.get_image_actor_scalars(self.image_store.images[self.image_store.reference].actor)
            self.bbox_window = BboxWindow(image)
            self.bbox_window.bbox_label.output_path_changed.connect(handle_output_path_change)
        else: utils.display_warning("Need to load an image first!")

    def set_bbox_color(self, color):
        self.bbox_store.bbox_actor.GetMapper().SetScalarVisibility(0)
        self.bbox_store.bbox_actor.GetProperty().SetColor(matplotlib.colors.to_rgb(color))

    def set_bbox_opacity(self, bbox_opacity: float):
        self.bbox_store.previous_opacity = self.bbox_store.bbox_opacity
        self.bbox_store.bbox_opacity = bbox_opacity
        self.bbox_store.bbox_actor.GetProperty().opacity = bbox_opacity
        
    def reset_bbox(self):
        if self.bbox_store.bbox_path:
            self.bbox_store.mirror_x = False
            self.bbox_store.mirror_y = False
            bbox = self.bbox_store.add_bbox(self.bbox_store.bbox_path, self.image_store.image_center, self.image_store.width, self.image_store.height)
            self.load_bbox(bbox)