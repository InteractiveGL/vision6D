import ast

import numpy as np
import PIL.Image

from PyQt5 import QtWidgets

from ..components import CameraStore
from ..components import ImageStore
from ..widgets import CalibrationPopWindow
from ..widgets import CameraPropsInputDialog

class CameraContainer:
    def __init__(self, plotter):
        self.plotter = plotter
        self.camera_store = CameraStore()
        self.image_store = ImageStore()

    def set_camera_props(self):
        self.camera_store.set_camera_intrinsics()
        self.camera_store.set_camera_extrinsics()
        self.plotter.camera = self.camera_store.camera.copy()

    def camera_calibrate(self):
        if self.image_store.image_path:
            original_image = np.array(PIL.Image.open(self.image_store.image_path), dtype='uint8')
            # make the the original image shape is [h, w, 3] to match with the rendered calibrated_image
            original_image = original_image[..., :3]
            if len(original_image.shape) == 2: original_image = original_image[..., None]
            if original_image.shape[-1] == 1: original_image = np.dstack((original_image, original_image, original_image))
            calibrated_image = np.array(self.image_store.render_image(self.plotter.camera.copy()), dtype='uint8')
            if original_image.shape != calibrated_image.shape:
                QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Original image shape is not equal to calibrated image shape!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return 0
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
        calibrate_pop = CalibrationPopWindow(calibrated_image, original_image)
        calibrate_pop.exec_()

    def set_camera(self):
        dialog = CameraPropsInputDialog(
            line1=("Fx", self.camera_store.fx), 
            line2=("Fy", self.camera_store.fy), 
            line3=("Cx", self.camera_store.cx), 
            line4=("Cy", self.camera_store.cy), 
            line5=("View Up", self.camera_store.cam_viewup), 
            line6=("Cam Position", self.camera_store.cam_position))
        if dialog.exec():
            fx, fy, cx, cy, cam_viewup, cam_position = dialog.getInputs()
            pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup, pre_cam_position = self.camera_store.fx, self.camera_store.fy, self.camera_store.cx, self.camera_store.cy, self.camera_store.cam_viewup, self.camera_store.cam_position
            if not (fx == '' or fy == '' or cx == '' or cy == '' or cam_viewup == '' or cam_position == ''):
                try:
                    self.camera_store.fx, self.camera_store.fy, self.camera_store.cx, self.camera_store.cy, self.camera_store.cam_viewup, self.camera_store.cam_position = ast.literal_eval(fx), ast.literal_eval(fy), ast.literal_eval(cx), ast.literal_eval(cy), ast.literal_eval(cam_viewup), ast.literal_eval(cam_position)
                    self.set_camera_props()
                except:
                    self.camera_store.fx, self.camera_store.fy, self.camera_store.cx, self.camera_store.cy, self.camera_store.cam_viewup, self.camera_store.cam_position = pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup, pre_cam_position
                    QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Error occured, check the format of the input values", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def reset_camera(self):
        self.plotter.camera = self.camera_store.camera.copy()

    def zoom_in(self):
        self.plotter.camera.zoom(2)

    def zoom_out(self):
        self.plotter.camera.zoom(0.5)
