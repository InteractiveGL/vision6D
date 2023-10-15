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
import numpy as np
import pyvista as pv

from PyQt5 import QtWidgets

from ..tools import utils
from ..components import CameraStore
from ..components import ImageStore
from ..components import BboxStore
from ..widgets import BboxWindow

class BboxContainer:
    def __init__(self, 
                plotter, 
                hintLabel,
                object_distance,
                track_actors_names, 
                add_button_actor_name, 
                check_button,
                output_text):

        self.plotter = plotter
        self.hintLabel = hintLabel
        self.object_distance = object_distance
        self.track_actors_names = track_actors_names
        self.add_button_actor_name = add_button_actor_name
        self.check_button = check_button
        self.output_text = output_text

        self.camera_store = CameraStore()
        self.image_store = ImageStore()
        self.bbox_store = BboxStore()

    def set_object_distance(self, object_distance):
        self.object_distance = object_distance

    def add_bbox_file(self, bbox_path='', prompt=False):
        if prompt:
            bbox_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.npy)") 
        if bbox_path:
            self.hintLabel.hide()
            self.add_bbox(bbox_path)

    def mirror_bbox(self, direction):
        if direction == 'x': self.bbox_store.mirror_x = not self.bbox_store.mirror_x
        elif direction == 'y': self.bbox_store.mirror_y = not self.bbox_store.mirror_y
        self.add_bbox(self.bbox_store.bbox_path)

    def load_bbox(self, bbox):
        # Add bbox surface object to the plot
        bbox_mesh = self.plotter.add_mesh(bbox, color="yellow", opacity=self.bbox_store.bbox_opacity, line_width=2)
        actor, _ = self.plotter.add_actor(bbox_mesh, pickable=True, name='bbox')
        self.bbox_store.bbox_actor = actor
                
    def add_bbox(self, bbox_source):
        if self.image_store.image_actor:
            bbox = self.bbox_store.add_bbox(bbox_source, self.image_store.width, self.image_store.height, self.object_distance)
            self.load_bbox(bbox)
            
            # Add remove current image to removeMenu
            if 'bbox' not in self.track_actors_names:
                self.track_actors_names.append('bbox')
                self.add_button_actor_name('bbox')
        else: utils.display_warning("Need to load an image first!")
    
    def draw_bbox(self):
        def handle_output_path_change(output_path):
            if output_path:
                self.bbox_store.bbox_path = output_path
                self.add_bbox(self.bbox_store.bbox_path)
        if self.image_store.image_actor:
            image = utils.get_image_actor_scalars(self.image_store.image_actor)
            self.bbox_window = BboxWindow(image)
            self.bbox_window.bbox_label.output_path_changed.connect(handle_output_path_change)
        else: utils.display_warning("Need to load an image first!")
        
    def reset_bbox(self):
        if self.bbox_store.bbox_path:
            self.bbox_store.mirror_x = False
            self.bbox_store.mirror_y = False
            bbox = self.bbox_store.add_bbox(self.bbox_store.bbox_path, self.image_store.width, self.image_store.height, self.object_distance)
            self.load_bbox(bbox)

    def export_bbox(self):
        if self.bbox_store.bbox_actor:
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Bbox Files (*.npy)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.png')
                # Store the transformed bbox actor if there is any transformation
                points = utils.get_bbox_actor_points(self.bbox_store.bbox_actor, self.bbox_store.image_center)
                np.save(output_path, points)
                self.output_text.append(f"-> Export Bbox points to:\n {output_path}")
            self.bbox_store.bbox_path = output_path
        else: utils.display_warning("Need to load a bounding box first!")