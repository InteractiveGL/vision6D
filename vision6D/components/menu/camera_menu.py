import numpy as np

class CameraMenu():

    def __init__(self, menu):

        # Save parameter
        self.menu = menu
        self.menu.addAction('Calibrate', self.camera_calibrate)
        self.menu.addAction('Reset Camera (d)', self.reset_camera)
        self.menu.addAction('Zoom In (x)', self.zoom_in)
        self.menu.addAction('Zoom Out (z)', self.zoom_out)

        QtWidgets.QShortcut(QtGui.QKeySequence("c"), self).activated.connect(self.reset_camera)
        QtWidgets.QShortcut(QtGui.QKeySequence("z"), self).activated.connect(self.zoom_out)
        QtWidgets.QShortcut(QtGui.QKeySequence("x"), self).activated.connect(self.zoom_in)

    def camera_calibrate(self):
        if self.image_path:
            original_image = np.array(PIL.Image.open(self.image_path), dtype='uint8')
            # make the the original image shape is [h, w, 3] to match with the rendered calibrated_image
            original_image = original_image[..., :3]
            if len(original_image.shape) == 2: original_image = original_image[..., None]
            if original_image.shape[-1] == 1: original_image = np.dstack((original_image, original_image, original_image))
            calibrated_image = np.array(self.render_image(self.plotter.camera.copy()), dtype='uint8')
            if original_image.shape != calibrated_image.shape:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Original image shape is not equal to calibrated image shape!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return 0
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
        calibrate_pop = CalibrationPopWindow(calibrated_image, original_image)
        calibrate_pop.exec_()

    def reset_camera(self):
        self.plotter.camera = self.camera.copy()

    def zoom_in(self):
        self.plotter.camera.zoom(2)

    def zoom_out(self):
        self.plotter.camera.zoom(0.5)
