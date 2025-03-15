# General import
import os
import cv2
import PIL
import numpy as np
from ..path import SAVE_ROOT
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# Qt5 import
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt

# self defined package import
np.set_printoptions(suppress=True)

class LiveWireLabel(QtWidgets.QLabel):
    output_path_changed = QtCore.pyqtSignal(str)
    def __init__(self, pixmap):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setPixmap(pixmap)
        self.setContentsMargins(0, 0, 0, 0)

        self.height = pixmap.height()
        self.width = pixmap.width()
        self.points = []
        self.path_segment_indices = []  # For tracking path segments
        self.output_path = None
        self.setMouseTracking(True)  # Enable mouse move events

        self.cv_image = self.qpixmap_to_cv_image(pixmap)
        self.compute_cost_image()
        self.cost_graph = None
        self.anchor_points = []
        self.live_wire_path = []

    def qpixmap_to_cv_image(self, pixmap):
        qimage = pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        return arr

    def compute_cost_image(self):
        # Convert image to grayscale
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        # Compute gradient magnitude using Sobel operator
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.hypot(gx, gy)
        # Normalize and invert the gradient to get the cost image
        self.cost_image = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        self.cost_image = 1 - self.cost_image  # Invert so that edges have lower cost

    def build_cost_graph(self):
        height, width = self.cost_image.shape
        num_nodes = height * width

        # Initialize lists for the sparse matrix
        data = []
        row_indices = []
        col_indices = []

        # Offsets for 8-connected neighbors
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),          (0, 1),
                (1, -1),  (1, 0),  (1, 1)]
        for y in range(height):
            for x in range(width):
                node_index = y * width + x
                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        neighbor_index = ny * width + nx
                        cost = (self.cost_image[y, x] + self.cost_image[ny, nx]) / 2
                        data.append(cost)
                        row_indices.append(node_index)
                        col_indices.append(neighbor_index)

        self.cost_graph = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))


    def compute_shortest_path(self, start_point, end_point):
        height, width = self.cost_image.shape

        # Define the search window
        window_size = 50  # Adjust as needed
        x_min = max(min(start_point.x(), end_point.x()) - window_size, 0)
        x_max = min(max(start_point.x(), end_point.x()) + window_size, width - 1)
        y_min = max(min(start_point.y(), end_point.y()) - window_size, 0)
        y_max = min(max(start_point.y(), end_point.y()) + window_size, height - 1)

        # Extract the sub-region from the cost image
        sub_cost_image = self.cost_image[y_min:y_max+1, x_min:x_max+1]
        sub_height, sub_width = sub_cost_image.shape
        num_nodes = sub_height * sub_width

        # Build the cost graph for the sub-region
        data = []
        row_indices = []
        col_indices = []

        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1),  (1, 0),  (1, 1)]
        for y in range(sub_height):
            for x in range(sub_width):
                node_index = y * sub_width + x
                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < sub_height and 0 <= nx < sub_width:
                        neighbor_index = ny * sub_width + nx
                        cost = (sub_cost_image[y, x] + sub_cost_image[ny, nx]) / 2
                        data.append(cost)
                        row_indices.append(node_index)
                        col_indices.append(neighbor_index)

        sub_cost_graph = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

        # Adjust start and end points relative to the sub-region
        start_x = start_point.x() - x_min
        start_y = start_point.y() - y_min
        end_x = end_point.x() - x_min
        end_y = end_point.y() - y_min
        start_index = start_y * sub_width + start_x
        end_index = end_y * sub_width + end_x

        # Compute Dijkstra's algorithm on the sub-region
        _, predecessors = dijkstra(csgraph=sub_cost_graph, directed=False, indices=start_index, return_predecessors=True)

        # Reconstruct the path
        path = []
        current = end_index
        while current != start_index and current != -9999:
            x = current % sub_width + x_min
            y = current // sub_width + y_min
            path.append(QtCore.QPoint(x, y))
            current = predecessors[current]
        path.append(QtCore.QPoint(start_point.x(), start_point.y()))
        path.reverse()
        return path

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S: self.save_mask()

    def save_mask(self):
        if len(self.points) < 3: QtWidgets.QMessageBox.warning(self, "Insufficient Points", "Please select at least 3 points."); return

        # Convert points to an array
        points = np.array([[p.x(), p.y()] for p in self.points], dtype=np.int32)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        user_input, ok = QtWidgets.QInputDialog.getText(self, 'Save Mask', 'Enter file name:')
        os.makedirs(SAVE_ROOT / "export_mask", exist_ok=True)
        if ok and user_input: self.output_path = SAVE_ROOT / "export_mask" / f'{user_input}.png'
        elif ok: self.output_path = SAVE_ROOT / "export_mask" / f'mask.png'
        else: return

        rendered_image = PIL.Image.fromarray(mask)
        rendered_image.save(self.output_path)
        self.output_path_changed.emit(str(self.output_path))
        self.window().close()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            self.anchor_points.append(pos)
            if len(self.anchor_points) > 1:
                path = self.compute_shortest_path(self.anchor_points[-2], self.anchor_points[-1])
                start_index = len(self.points)
                self.points.extend(path)
                self.path_segment_indices.append(start_index)
            else:
                self.points.append(pos)
                self.path_segment_indices.append(0)
            self.update()
        elif event.button() == Qt.RightButton and self.anchor_points:
            self.anchor_points.pop()
            if self.path_segment_indices:
                start_index = self.path_segment_indices.pop()
                self.points = self.points[:start_index]
            self.update()

    def mouseMoveEvent(self, event):
        if self.anchor_points:
            self.current_position = event.pos()
            self.live_wire_path = self.compute_shortest_path(self.anchor_points[-1], self.current_position)
            self.update()
        else:
            self.live_wire_path = []
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 255), 3))

        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                painter.drawLine(self.points[i], self.points[i+1])

        if self.live_wire_path:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 3, Qt.SolidLine))
            for i in range(len(self.live_wire_path) - 1):
                painter.drawLine(self.live_wire_path[i], self.live_wire_path[i+1])

        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 3))
        for point in self.anchor_points:
            painter.drawEllipse(point, 4, 4)

class LiveWireWindow(QtWidgets.QWidget):
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

        self.mask_label = LiveWireLabel(pixmap)
        layout.addWidget(self.mask_label)
        self.setLayout(layout)
        self.show()