'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: draw_bbox_window.py
@time: 2023-07-03 20:32
@desc: create the window for bounding box labeling/drawing
'''

import pathlib

import numpy as np
import PIL.Image

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPen, QPainter, QColor

class BboxLabel(QtWidgets.QLabel):
    output_path_changed = QtCore.pyqtSignal(str)
    def __init__(self, pixmap):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.pixmap = pixmap
        self.setPixmap(self.pixmap)
        self.setContentsMargins(0, 0, 0, 0)

        self.points = QtGui.QPolygon()
        self.label = QLabel(self)
        self.label.setStyleSheet("color: rgb(255, 255, 0); background-color: transparent; padding: 5px")
        self.label.setFixedWidth(self.pixmap.width() // 2)
        self.label.hide()
        
        self.output_path = None
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S: self.save_bbox()
            
    def save_bbox(self):
        if hasattr(self, "rect_start") and hasattr(self, "rect_end"):
            # Define the rectangle's four corners
            tl_x = min(self.rect_start.x(), self.rect_end.x())  # top-left x
            tl_y = min(self.rect_start.y(), self.rect_end.y())  # top-left y
            br_x = max(self.rect_start.x(), self.rect_end.x())  # bottom-right x
            br_y = max(self.rect_start.y(), self.rect_end.y())  # bottom-right y

            # Create the points array
            points = np.array([tl_x, tl_y, br_x, br_y])

            self.output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Bbox Files (*.npy)")
            if self.output_path:
                if pathlib.Path(self.output_path).suffix == '': self.output_path = str(pathlib.Path(self.output_path).parent / (pathlib.Path(self.output_path).stem + '.png'))
                np.save(self.output_path, points)
                self.output_path_changed.emit(self.output_path)

    def get_normalized_rect(self):
        left = min(self.rect_start.x(), self.rect_end.x())
        top = min(self.rect_start.y(), self.rect_end.y())
        width = abs(self.rect_start.x() - self.rect_end.x())
        height = abs(self.rect_start.y() - self.rect_end.y())
        return QRect(left, top, width, height)

    def move_label(self):
        # Update the label position to be at the bottom right of the bounding box
        self.label.move(min(self.rect_start.x(), self.rect_end.x()), min(self.rect_start.y(), self.rect_end.y()))

        # Update the label text with the width and height values
        width = abs(self.rect_start.x() - self.rect_end.x())
        height = abs(self.rect_start.y() - self.rect_end.y())
        self.label.setText(f"Width: {width}, Height: {height}")

        # Show the label
        self.label.show()

    def mousePressEvent(self, event):
        self.label.hide()
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            self.rect_start = pos
            self.rect_end = pos
            self.update()
        elif event.button() == Qt.MiddleButton:
            self.drag_start = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.rect_end = event.pos()
            self.update()
            self.move_label()
            
        elif event.buttons() == Qt.MiddleButton:
            delta = event.pos() - self.drag_start
            self.rect_start += delta
            self.rect_end += delta
            self.drag_start = event.pos()
            self.update()
            self.move_label()

    def mouseReleaseEvent(self, event):
            pos = event.pos()
            rect = self.get_normalized_rect()
            if rect.contains(pos): self.drag_start = pos

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)

        if hasattr(self, "rect_start") and hasattr(self, "rect_end"):
            rect = self.get_normalized_rect()
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawRect(rect)

class BboxWindow(QtWidgets.QWidget):
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

        self.bbox_label = BboxLabel(pixmap)
        layout.addWidget(self.bbox_label)
        self.setLayout(layout)
        self.show()

