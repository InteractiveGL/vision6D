'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: camera_container.py
@time: 2023-07-03 20:25
@desc: create container for camera related actions in application
'''

import ast
import math
import pickle
import pathlib

import numpy as np
import PIL.Image

from PyQt5 import QtWidgets

from ..tools import utils
from ..components import CameraStore
from ..components import ImageStore
from ..widgets import CalibrationDialog
from ..widgets import CameraPropsInputDialog

class CameraContainer:
    def __init__(self, plotter):
        self.plotter = plotter
        self.camera_store = CameraStore()
        self.image_store = ImageStore()

    def reset_camera(self):
        self.plotter.camera = self.camera_store.camera.copy()

    def set_camera_props(self):
        self.camera_store.set_camera_intrinsics()
        self.camera_store.set_camera_extrinsics()
        self.reset_camera()

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
            line1=("Fx", self.camera_store.fx), 
            line2=("Fy", self.camera_store.fy), 
            line3=("Cx", self.camera_store.cx), 
            line4=("Cy", self.camera_store.cy), 
            line5=("View Up", self.camera_store.cam_viewup))
        if dialog.exec():
            fx, fy, cx, cy, cam_viewup = dialog.getInputs()
            pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup = self.camera_store.fx, self.camera_store.fy, self.camera_store.cx, self.camera_store.cy, self.camera_store.cam_viewup
            if not (fx == '' or fy == '' or cx == '' or cy == '' or cam_viewup == ''):
                try:
                    self.camera_store.fx, self.camera_store.fy, self.camera_store.cx, self.camera_store.cy, self.camera_store.cam_viewup = ast.literal_eval(fx), ast.literal_eval(fy), ast.literal_eval(cx), ast.literal_eval(cy), ast.literal_eval(cam_viewup)
                    self.set_camera_props()
                except:
                    self.camera_store.fx, self.camera_store.fy, self.camera_store.cx, self.camera_store.cy, self.camera_store.cam_viewup = pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup
                    utils.display_warning("Error occured, check the format of the input values")

    def zoom_in(self):
        self.plotter.camera.zoom(2)

    def zoom_out(self):
        self.plotter.camera.zoom(0.5)

    def export_camera_info(self):
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Camera Info Files (*.pkl)")
        if output_path:
            if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.pkl')
            camera_intrinsics = self.camera_store.camera_intrinsics.astype('float32')
            focal_length = (1080 / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
            camera_intrinsics[0, 0] = focal_length
            camera_intrinsics[1, 1] = focal_length
            camera_info = {'camera_intrinsics': camera_intrinsics}
            with open(output_path,"wb") as f: pickle.dump(camera_info, f)
