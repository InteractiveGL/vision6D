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

from ..tools import utils
from ..components import ImageStore
from ..components import CameraStore
from ..widgets import CalibrationDialog, CameraPropsInputDialog
        
class ImageContainer:
    def __init__(self, plotter):
          
        self.plotter = plotter
        self.image_store = ImageStore()
        self.camera_store = CameraStore(plotter)

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
            line1=("Fx", self.image_store.images[self.image_store.reference].fx), 
            line2=("Fy", self.image_store.images[self.image_store.reference].fy), 
            line3=("Cx", self.image_store.images[self.image_store.reference].cx), 
            line4=("Cy", self.image_store.images[self.image_store.reference].cy), 
            line5=("View Up", self.image_store.images[self.image_store.reference].cam_viewup))
        if dialog.exec():
            fx, fy, cx, cy, cam_viewup = dialog.getInputs()
            pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup = self.image_store.images[self.image_store.reference].fx, self.image_store.images[self.image_store.reference].fy, self.image_store.images[self.image_store.reference].cx, self.image_store.images[self.image_store.reference].cy, self.image_store.images[self.image_store.reference].cam_viewup
            if not (fx == '' or fy == '' or cx == '' or cy == '' or cam_viewup == ''):
                try:
                    self.image_store.images[self.image_store.reference].fx, self.image_store.images[self.image_store.reference].fy, self.image_store.images[self.image_store.reference].cx, self.image_store.images[self.image_store.reference].cy, self.image_store.images[self.image_store.reference].cam_viewup = ast.literal_eval(fx), ast.literal_eval(fy), ast.literal_eval(cx), ast.literal_eval(cy), ast.literal_eval(cam_viewup)
                    self.camera_store.set_camera_intrinsics()
                    self.camera_store.set_camera_extrinsics()
                    self.image_store.images[self.image_store.reference].image_pv.translate(self.image_store.images[self.image_store.reference].image_center, inplace=True) # reset the image position
                    self.image_store.images[self.image_store.reference].distance2camera = self.image_store.images[self.image_store.reference].fy # set the frame distance to the camera
                    self.image_store.images[self.image_store.reference].cx_offset = (self.image_store.images[self.image_store.reference].cx - (self.image_store.images[self.image_store.reference].width / 2.0))
                    self.image_store.images[self.image_store.reference].cy_offset = (self.image_store.images[self.image_store.reference].cy - (self.image_store.images[self.image_store.reference].height / 2.0))
                    print(f"Image New Origin: {self.image_store.images[self.image_store.reference].cx_offset, self.image_store.images[self.image_store.reference].cy_offset}")
                    self.image_store.images[self.image_store.reference].image_center = np.array([self.image_store.images[self.image_store.reference].cx_offset, self.image_store.images[self.image_store.reference].cy_offset, -self.image_store.images[self.image_store.reference].fy])
                    self.image_store.images[self.image_store.reference].image_pv.translate(self.image_store.images[self.image_store.reference].image_center, inplace=True) # move the image to the camera distance
                    self.camera_store.reset_camera()
                except:
                    self.image_store.images[self.image_store.reference].fx, self.image_store.images[self.image_store.reference].fy, self.image_store.images[self.image_store.reference].cx, self.image_store.images[self.image_store.reference].cy, self.image_store.images[self.image_store.reference].cam_viewup = pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup
                    utils.display_warning("Error occured, check the format of the input values")

    #^ Image related
    def mirror_image(self, direction):
        if direction == 'x': self.image_store.mirror_x = not self.image_store.mirror_x
        elif direction == 'y': self.image_store.mirror_y = not self.image_store.mirror_y
        self.add_image(self.image_store.image_source)

    def add_image(self, image_source):
        # self.set_camera()
        image_data = self.image_store.add_image(image_source)
        if image_data.channel == 1: 
            image = self.plotter.add_mesh(image_data.image_pv, cmap='gray', opacity=image_data.opacity, name=image_data.name)
        else: 
            image = self.plotter.add_mesh(image_data.image_pv, rgb=True, opacity=image_data.opacity, name=image_data.name)
        actor, _ = self.plotter.add_actor(image, pickable=False, name=image_data.name)
        image_data.actor = actor
        return image_data
                                          
    def set_image_opacity(self, name: str, opacity: float):
        image_data = self.image_store.images[name]
        image_data.previous_opacity = image_data.opacity
        image_data.opacity = opacity
        image_data.actor.GetProperty().opacity = opacity