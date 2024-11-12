# General import
import os
import cv2
import PIL.Image
import numpy as np

# Qt5 import
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from ..path import SAVE_ROOT

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
        
        user_input, ok = QtWidgets.QInputDialog.getText(self, 'Save Mask', 'Enter file name:')
        os.makedirs(SAVE_ROOT / "export_mask", exist_ok=True)
        if ok and user_input: self.output_path = SAVE_ROOT / "export_mask" / f'{user_input}.png'
        elif ok: self.output_path = SAVE_ROOT / "export_mask" / f'mask.png'
        else: return

        rendered_image = PIL.Image.fromarray(image)
        rendered_image.save(self.output_path)
        self.output_path_changed.emit(str(self.output_path))
        self.window().close()

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