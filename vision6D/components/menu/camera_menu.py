from PyQt5 import QtWidgets

from ...widgets import CalibrationPopWindow
from ...stores import PlotStore
from ...stores import ImageStore

class CameraMenu():

    def __init__(self):

        self.plot_store = PlotStore()
        self.image_store = ImageStore()

    def calibrate(self):
        if self.image_store.image_path: 
            original_image, calibrated_image = self.image_store.calibrate_image()
            if original_image.shape == calibrated_image.shape:
                calibrate_pop = CalibrationPopWindow(calibrated_image, original_image)
                calibrate_pop.exec_()
                return ''
            else:
                return "Original image shape is not equal to calibrated image shape!"
        else:
            return "Need to load an image first!"

