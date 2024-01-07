'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: draw_sam_window.py
@time: 2024-01-06 18:37
@desc: create the window for bounding box labeling/drawing
'''

import pathlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image

from segment_anything import SamPredictor, sam_model_registry

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPen, QPainter, QColor

from ..path import MODEL_PATH

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

class SamLabel(QtWidgets.QLabel):
    output_path_changed = QtCore.pyqtSignal(str)
    def __init__(self, image_source, pixmap):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.image_source = image_source
        self.pixmap = pixmap
        self.setPixmap(self.pixmap)
        self.setContentsMargins(0, 0, 0, 0)

        self.points = QtGui.QPolygon()
        self.label = QLabel(self)
        self.label.setStyleSheet("color: rgb(255, 255, 0); background-color: transparent; padding: 5px")
        self.label.setFixedWidth(self.pixmap.width() // 2)
        self.label.hide()
        
        self.rect_start = None
        self.rect_end = None
        self.output_path = None
        self.smoothed_points = QtGui.QPolygon()
        self.selected_point_index = None
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S: self.save_bbox()
            
    def save_bbox(self):
        if self.rect_start is not None and self.rect_end is not None:
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

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.rect_start = None
            self.rect_end = None
            if len(self.smoothed_points) > 0:
                for i, point in enumerate(self.smoothed_points):
                    if (point - event.pos()).manhattanLength() < 10:  # 10 is the sensitivity, adjust as needed
                        self.selected_point_index = i
                        break
            else:
                pos = event.pos()
                self.rect_start = pos
                self.rect_end = pos
                
        elif event.button() == Qt.RightButton:
            self.label.hide()
            self.smoothed_points.clear()
            self.selected_point_index = None
        
        self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            if self.selected_point_index is None:
                self.rect_end = event.pos()
            elif (self.smoothed_points[self.selected_point_index] - event.pos()).manhattanLength() < 10:
                self.smoothed_points[self.selected_point_index] = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        if self.rect_start is not None and self.rect_end is not None:
            pos = event.pos()
            rect = self.get_normalized_rect()
            if rect.contains(pos): self.drag_start = pos
            self.sam_prediction(rect)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)

        if len(self.smoothed_points) == 0 and self.rect_start is not None and self.rect_end is not None:
            rect = self.get_normalized_rect()
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawRect(rect)

        if len(self.smoothed_points) > 0:
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 0)))
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            for point in self.smoothed_points:
                painter.drawEllipse(point, 4, 4)
            painter.drawPolygon(self.smoothed_points)

    def find_large_contour(self, mask):
        num_labels, labels_im = cv2.connectedComponents(mask)
        label_sizes = [(labels_im == label).sum() for label in range(1, num_labels)]
        max_label = np.argmax(label_sizes) + 1
        largest_component = np.zeros_like(labels_im, dtype=np.uint8)
        largest_component[labels_im == max_label] = 1
        contours, _ = cv2.findContours(largest_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = contours[0].squeeze()
        epsilon = 0.01 * cv2.arcLength(points, True)
        smoothed_points = cv2.approxPolyDP(points, epsilon, True)
        for point in smoothed_points.squeeze(): 
            self.smoothed_points.append(QtCore.QPoint(point[0], point[1]))
            
    def sam_prediction(self, rect):
        sam = sam_model_registry["vit_h"](checkpoint=str(MODEL_PATH / "sam_vit_h_4b8939.pth"))
        sam.to(device="cuda")
        predictor = SamPredictor(sam)
        predictor.set_image(self.image_source)
        input_box = np.array([rect.x(), rect.y(), rect.x()+rect.width(), rect.y()+rect.height()])
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        self.find_large_contour(masks.squeeze().astype(np.uint8))

class SamWindow(QtWidgets.QWidget):
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

        self.mask_label = SamLabel(image_source, pixmap)
        layout.addWidget(self.mask_label)
        self.setLayout(layout)
        self.show()

