'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: image_container.py
@time: 2023-07-03 20:26
@desc: create container for image related actions in application
'''
import ast
import math
import pickle
import pathlib
import numpy as np
import PIL.Image
from PyQt5 import QtWidgets

from ..tools import utils
from ..components import ImageStore
from ..widgets import CalibrationDialog, CameraPropsInputDialog

class ImageContainer:
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

        self.image_store = ImageStore()

    def set_object_distance(self, object_distance):
        self.object_distance = object_distance

    #^ Camera related
    def camera_calibrate(self):
        if self.image_store.image_path:
            original_image = np.array(PIL.Image.open(self.image_store.image_path), dtype='uint8')
            # make the the original image shape is [h, w, 3] to match with the rendered calibrated_image
            original_image = original_image[..., :3]
            if len(original_image.shape) == 2: original_image = original_image[..., None]
            if original_image.shape[-1] == 1: original_image = np.dstack((original_image, original_image, original_image))
            calibrated_image = np.array(self.image_store.render_image(self.plotter.camera.copy()), dtype='uint8')
            if original_image.shape != calibrated_image.shape:
                utils.display_warning("Original image shape is not equal to calibrated image shape!")
            else: CalibrationDialog(calibrated_image, original_image).exec_()
        else: utils.display_warning("Need to load an image first!")

    def set_camera(self):
        dialog = CameraPropsInputDialog(
            line1=("Fx", self.image_store.fx), 
            line2=("Fy", self.image_store.fy), 
            line3=("Cx", self.image_store.cx), 
            line4=("Cy", self.image_store.cy), 
            line5=("View Up", self.image_store.cam_viewup))
        if dialog.exec():
            fx, fy, cx, cy, cam_viewup = dialog.getInputs()
            pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup = self.image_store.fx, self.image_store.fy, self.image_store.cx, self.image_store.cy, self.image_store.cam_viewup
            if not (fx == '' or fy == '' or cx == '' or cy == '' or cam_viewup == ''):
                try:
                    self.image_store.fx, self.image_store.fy, self.image_store.cx, self.image_store.cy, self.image_store.cam_viewup = ast.literal_eval(fx), ast.literal_eval(fy), ast.literal_eval(cx), ast.literal_eval(cy), ast.literal_eval(cam_viewup)
                    self.image_store.set_camera_props()
                except:
                    self.image_store.fx, self.image_store.fy, self.image_store.cx, self.image_store.cy, self.image_store.cam_viewup = pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup
                    utils.display_warning("Error occured, check the format of the input values")

    def export_camera_info(self):
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Camera Info Files (*.pkl)")
        if output_path:
            if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.pkl')
            camera_intrinsics = self.image_store.camera_intrinsics.astype('float32')
            if self.image_store.height:
                focal_length = (self.image_store.height / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
                camera_intrinsics[0, 0] = focal_length
                camera_intrinsics[1, 1] = focal_length
            camera_info = {'camera_intrinsics': camera_intrinsics}
            with open(output_path,"wb") as f: pickle.dump(camera_info, f)
            self.output_text.append(f"-> Export camera info to:\n {output_path}")

    #^ Image related
    def add_image_file(self, image_path='', prompt=False):
        if prompt:
            image_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)")
        if image_path:
            self.hintLabel.hide()
            self.add_image(image_path)

    def mirror_image(self, direction):
        if direction == 'x': self.image_store.mirror_x = not self.image_store.mirror_x
        elif direction == 'y': self.image_store.mirror_y = not self.image_store.mirror_y
        self.add_image(self.image_store.image_source)

    def add_image(self, image_source):
        image, _, channel = self.image_store.add_image(image_source, self.object_distance)
        if channel == 1: image = self.plotter.add_mesh(image, cmap='gray', opacity=self.image_store.image_opacity, name='image')
        else: image = self.plotter.add_mesh(image, rgb=True, opacity=self.image_store.image_opacity, name='image')
        actor, _ = self.plotter.add_actor(image, pickable=False, name='image')
        self.image_store.image_actor = actor
        # add remove current image to removeMenu
        if 'image' not in self.track_actors_names:
            self.track_actors_names.append('image')
            self.add_button_actor_name('image')
                                          
    def set_image_opacity(self, image_opacity: float):
        self.image_store.image_opacity = image_opacity
        self.image_store.image_actor.GetProperty().opacity = image_opacity

    def toggle_image_opacity(self, up):
        change = 0.05
        if not up: change *= -1
        self.image_store.update_opacity(change)
        self.check_button(name='image')

    def export_image(self):
        if self.image_store.image_actor:
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Image Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.png')
                image_rendered = self.image_store.render_image(camera=self.plotter.camera.copy())
                rendered_image = PIL.Image.fromarray(image_rendered)
                rendered_image.save(output_path)
                self.output_text.append(f"-> Export image render to:\n {output_path}")
            self.image_store.image_path = output_path
        else: utils.display_warning("Need to load an image first!")