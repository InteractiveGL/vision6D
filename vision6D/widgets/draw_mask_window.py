'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: draw_mask_window.py
@time: 2023-07-03 20:32
@desc: create the window for mask labeling/drawing
'''

# General import
import numpy as np
import cv2
import pathlib
import PIL.Image

# Qt5 import
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt

# self defined package import
np.set_printoptions(suppress=True)

class MaskLabel(QtWidgets.QLabel):
    output_path_changed = QtCore.pyqtSignal(str)
    def __init__(self, pixmap):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setPixmap(pixmap)
        self.setContentsMargins(0, 0, 0, 0)
        
        self.height = pixmap.height()
        self.width = pixmap.width()
        self.points = QtGui.QPolygon()
        self.output_path = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S: self.save_mask()

    def save_mask(self):
        points = []
        for point in self.points:
            coord = [point.x(), point.y()]
            points.append(coord)

        points = np.array(points).astype('int32')
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        image = cv2.fillPoly(mask, [points], 255)
        self.output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Mask Files (*.png)")
        if self.output_path:
            if pathlib.Path(self.output_path).suffix == '': self.output_path = str(pathlib.Path(self.output_path).parent / (pathlib.Path(self.output_path).stem + '.png'))
            rendered_image = PIL.Image.fromarray(image)
            rendered_image.save(self.output_path)
            self.output_path_changed.emit(self.output_path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.points.append(event.pos())
        elif event.button() == Qt.RightButton and not self.points.isEmpty():
            self.points.remove(self.points.size()-1)
        self.update()
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QColor(0, 0, 255))
        
        if not self.points.isEmpty():
            complete_points = QtGui.QPolygon(self.points)
            if (QtCore.QLineF(self.points.first(), self.points.last()).length() < 10) and (self.points.first() != self.points.last()):
                complete_points.append(self.points.first())
            # Draw the first point with a larger radius
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255)))
            painter.drawEllipse(self.points.point(0), 2, 2)
            painter.drawPolyline(complete_points)

class MaskWindow(QtWidgets.QWidget):
    def __init__(self, image_source):
        super().__init__()
        image = QtGui.QImage(image_source.tobytes(), image_source.shape[1], image_source.shape[0], image_source.shape[2]*image_source.shape[1], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.setFixedSize(pixmap.size())

        layout = QtWidgets.QVBoxLayout()

        #* setContentsMargins sets the width of the outside border around the layout
        layout.setContentsMargins(0, 0, 0, 0)
        #* setSpacing sets the width of the inside border between widgets in the layout.
        layout.setSpacing(0)
        #* Both are set to zero to eliminate any space between the widgets and the layout border.

        self.mask_label = MaskLabel(pixmap)
        layout.addWidget(self.mask_label)
        self.setLayout(layout)
        self.show()