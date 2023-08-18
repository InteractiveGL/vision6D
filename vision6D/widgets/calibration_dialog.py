'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: calibration_dialog.py
@time: 2023-07-03 20:29
@desc: pop window for camera calibration
'''

# General import
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Qt5 import
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt

# self defined package import
np.set_printoptions(suppress=True)
    
class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, calibrated_image, original_image, parent=None):
        super(CalibrationDialog, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.calibrated_image = calibrated_image
        self.original_image = original_image

        self.setWindowTitle("Vision6D")

        # Set the size for display
        if self.original_image.shape[1] > 960 and self.original_image.shape[1] > 540:
            size = int(self.original_image.shape[1] // 2), int(self.original_image.shape[0] // 2)
        else: size = int(self.original_image.shape[1]), int(self.original_image.shape[0])
        
        overall = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QHBoxLayout()

        vbox1layout = QtWidgets.QVBoxLayout()
        label1 = QtWidgets.QLabel("Calibrated image", self)
        label1.setAlignment(Qt.AlignCenter)
        pixmap_label1 = QtWidgets.QLabel(self)

        qimage1 = self.numpy_to_qimage(self.calibrated_image)
        pixmap1 = QtGui.QPixmap.fromImage(qimage1).scaled(*size, Qt.KeepAspectRatio)
        
        pixmap_label1.setPixmap(pixmap1)
        pixmap_label1.setAlignment(Qt.AlignCenter)
        pixmap_label1.setFixedSize(*size)
        
        vbox1layout.addWidget(label1)
        vbox1layout.addWidget(pixmap_label1)

        vbox2layout = QtWidgets.QVBoxLayout()
        label2 = QtWidgets.QLabel("Original image", self)
        label2.setAlignment(Qt.AlignCenter)
        pixmap_label2 = QtWidgets.QLabel(self)

        qimage2 = self.numpy_to_qimage(self.original_image)
        pixmap2 = QtGui.QPixmap.fromImage(qimage2).scaled(*size, Qt.KeepAspectRatio)
        
        pixmap_label2.setPixmap(pixmap2)
        pixmap_label2.setAlignment(Qt.AlignCenter)
        pixmap_label2.setFixedSize(*size)
        
        vbox2layout.addWidget(label2)
        vbox2layout.addWidget(pixmap_label2)

        layout.addLayout(vbox1layout)
        layout.addLayout(vbox2layout)

        psnr, ssim = self.calculate_similarity()
        similarity_label = QtWidgets.QLabel(f"PSNR is {psnr} and SSIM is {ssim}", self)
        similarity_label.setAlignment(Qt.AlignCenter)

        overall.addLayout(layout)
        overall.addWidget(similarity_label)

        self.setLayout(overall)

    def numpy_to_qimage(self, array):
        height, width, channel = array.shape
        #* bytesPerline is very important
        bytesPerLine = channel * width
        return QtGui.QImage(array.tobytes(), width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

    def calculate_similarity(self):
        psnr = peak_signal_noise_ratio(self.calibrated_image, self.original_image, data_range=255)
        ssim = structural_similarity(self.calibrated_image, self.original_image, data_range=255, channel_axis=2)
        return psnr, ssim