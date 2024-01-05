'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: mask_container.py
@time: 2023-07-03 20:26
@desc: create container for mask related actions in application
'''
import os
import ast
import pathlib

import numpy as np
import PIL.Image
import matplotlib.colors

from PyQt5 import QtWidgets

from ..path import PKG_ROOT
from ..tools import utils, exception
from ..components import CameraStore
from ..components import ImageStore
from ..components import MaskStore
from ..widgets import MaskWindow, GetMaskDialog

class MaskContainer:
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
        self.mask_store = MaskStore()

    def set_object_distance(self, object_distance):
        self.object_distance = object_distance

    def add_mask_file(self, mask_path='', prompt=False):
        if prompt:
            mask_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.npy *.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)") 
        if mask_path:
            self.hintLabel.hide()
            self.add_mask(mask_path)

    def set_mask(self):
        get_mask_dialog = GetMaskDialog()
        res = get_mask_dialog.exec_()
        if res == QtWidgets.QDialog.Accepted:
            if get_mask_dialog.mask_path: self.add_mask_file(get_mask_dialog.mask_path)
            else:
                user_text = get_mask_dialog.get_text()
                points = exception.set_data_format(user_text)
                if points is not None:
                    if points.shape[1] == 2:
                        os.makedirs(PKG_ROOT.parent / "output", exist_ok=True)
                        os.makedirs(PKG_ROOT.parent / "output" / "mask_points", exist_ok=True)
                        if self.image_store.image_path: mask_path = PKG_ROOT.parent / "output" / "mask_points" / f"{pathlib.Path(self.image_store.image_path).stem}.npy"
                        else: mask_path = PKG_ROOT.parent / "output" / "mask_points" / "mask_points.npy"
                        np.save(mask_path, points)
                        self.add_mask_file(mask_path)
                    else:
                        utils.display_warning("It needs to be a n by 2 matrix")

    def mirror_mask(self, direction):
        if direction == 'x': self.mask_store.mirror_x = not self.mask_store.mirror_x
        elif direction == 'y': self.mask_store.mirror_y = not self.mask_store.mirror_y
        self.add_mask(self.mask_store.mask_path)

    def load_mask(self, mask_surface):
        # Add mask surface object to the plot
        mask_mesh = self.plotter.add_mesh(mask_surface, color=self.mask_store.color, style='surface', opacity=self.mask_store.mask_opacity)
        actor, _ = self.plotter.add_actor(mask_mesh, pickable=True, name='mask')
        self.mask_store.mask_actor = actor
        
    def add_mask(self, mask_source):
        mask_surface = self.mask_store.add_mask(mask_source, self.object_distance, self.plotter.main_window.window_size)
        self.load_mask(mask_surface)
        
        # Add remove current image to removeMenu
        if 'mask' not in self.track_actors_names:
            self.track_actors_names.append('mask')
            self.add_button_actor_name('mask')
    
    def reset_mask(self):
        if self.mask_store.mask_path:
            self.mask_store.mirror_x = False
            self.mask_store.mirror_y = False
            mask_surface = self.mask_store.add_mask(self.mask_store.mask_path, self.object_distance, self.plotter.main_window.window_size)
            self.load_mask(mask_surface)

    def set_mask_opacity(self, mask_opacity: float):
        self.mask_store.mask_opacity = mask_opacity
        self.mask_store.mask_actor.GetProperty().opacity = mask_opacity

    def set_mask_color(self, color):
        self.mask_store.mask_actor.GetMapper().SetScalarVisibility(0)
        self.mask_store.mask_actor.GetProperty().SetColor(matplotlib.colors.to_rgb(color))
    
    def toggle_mask_opacity(self, up):
        change = 0.05
        if not up: change *= -1
        self.mask_store.update_opacity(change)
        self.check_button(name='mask')
    
    def draw_mask(self):
        def handle_output_path_change(output_path):
            if output_path:
                self.mask_store.mask_path = output_path
                self.add_mask(self.mask_store.mask_path)
        if self.image_store.image_actor:
            image = utils.get_image_actor_scalars(self.image_store.image_actor)
            self.mask_window = MaskWindow(image)
            self.mask_window.mask_label.output_path_changed.connect(handle_output_path_change)
        else: utils.display_warning("Need to load an image first!")

    def export_mask(self):
        if self.mask_store.mask_actor:
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Mask Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.png')
                # Store the transformed mask actor if there is any transformation
                mask_surface = self.mask_store.update_mask()
                self.load_mask(mask_surface)
                image = self.mask_store.render_mask(camera=self.plotter.camera.copy())
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.output_text.append(f"-> Export mask render to:\n {output_path}")
            self.mask_store.mask_path = output_path
        else: utils.display_warning("Need to load a mask first!")