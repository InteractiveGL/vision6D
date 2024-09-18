import numpy as np
import cv2
# Qt5 import
from PyQt5 import QtWidgets, QtGui, QtCore
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt
from ..tools import utils

class PnPLabel(QtWidgets.QLabel):
    def __init__(self, pixmap):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setPixmap(pixmap)
        self.setContentsMargins(0, 0, 0, 0)
        
        self.height = pixmap.height()
        self.width = pixmap.width()
        self.points = QtGui.QPolygon()

    def get_2d_points(self):
        points = []
        for point in self.points: points.append([point.x(), point.y()])
        points = np.array(points).astype('int32')
        return points

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self.points.append(event.pos())
        elif event.button() == Qt.RightButton and not self.points.isEmpty(): self.points.remove(self.points.size()-1)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QColor(255, 0, 0))
        
        if not self.points.isEmpty():
            for i in range(self.points.size()):
                point = self.points.point(i)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)))
                painter.drawEllipse(point, 4, 4)
                painter.setPen(QtGui.QColor(255, 0, 0))
                painter.drawText(point.x() + 10, point.y() + 10, str(i + 1))

class PnPWindow(QtWidgets.QWidget):
    transformation_matrix_computed = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, image_source, mesh_data, camera_intrinsics):
        super().__init__()

        self.camera_intrinsics = camera_intrinsics
        self.picked_3d_points = []
        self.point_labels = []
        
        # Set window size and layout
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        content_layout = QtWidgets.QHBoxLayout()

        # Left panel: 2D image display
        image = QtGui.QImage(image_source.tobytes(), image_source.shape[1], image_source.shape[0], image_source.shape[2] * image_source.shape[1], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        image_layout = QtWidgets.QVBoxLayout()
        self.pnp_label = PnPLabel(pixmap)
        image_layout.addWidget(self.pnp_label)
        left_panel = QtWidgets.QWidget()
        left_panel.setLayout(image_layout)
        content_layout.addWidget(left_panel)

        # Right panel: 3D visualization using PyVista
        self.pv_widget = QtInteractor(self)
        self.plot_3d_model(mesh_data)
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.pv_widget)
        right_panel.setLayout(right_layout)
        content_layout.addWidget(right_panel)

        # Add the content (2D image + 3D view) to the main vertical layout
        layout.addLayout(content_layout)

        # Add the submit button at the bottom
        submit_button = QtWidgets.QPushButton("(At least four points to perform PnP registration) Submit")
        submit_button.clicked.connect(self.submit_to_pnp_register)
        layout.addWidget(submit_button)
        self.show()

    def submit_to_pnp_register(self):
        picked_2d_points = self.pnp_label.get_2d_points()
        if len(picked_2d_points) < 4 or len(picked_2d_points) != len(self.picked_3d_points):
            utils.display_warning("Please select at least 4 points to perform PnP algorithm and the picked 2D points should match the picked 3D points.")
        else:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                objectPoints=np.array(self.picked_3d_points, dtype=np.float32),
                imagePoints=np.array(picked_2d_points, dtype=np.float32),
                cameraMatrix=self.camera_intrinsics,
                distCoeffs=np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_EPNP
            )
            if success:
                transformation_matrix = np.eye(4)
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                transformation_matrix[:3, :3] = rotation_matrix
                transformation_matrix[:3, 3] = translation_vector.reshape(3)
                self.transformation_matrix_computed.emit(transformation_matrix)
                self.close()
            else: utils.display_warning("Pose estimation failed. Please try to select different non-coplanar points.")

    def point_picking_callback(self, point):
        self.picked_3d_points.append(point)
        label_text = str(len(self.picked_3d_points))
        label_actor = self.pv_widget.add_point_labels([point], [label_text], show_points=False, font_size=14, text_color="red", shape='rect', shape_opacity=1, pickable=False, reset_camera=False)
        self.point_labels.append(label_actor)

    def right_click_callback(self, *args):
        if self.picked_3d_points:
            self.picked_3d_points.pop()
            self.pv_widget.renderer.RemoveActor(self.point_labels.pop())
            self.pv_widget.render()

    def plot_3d_model(self, mesh_data):
        self.mesh_data = mesh_data
        self.mesh_actor = self.pv_widget.add_mesh(mesh_data)
        self.pv_widget.enable_surface_point_picking(callback=self.point_picking_callback, show_message=False, show_point=False, left_clicking=True, color='red')
        self.pv_widget.iren.add_observer("RightButtonPressEvent", self.right_click_callback)
        self.pv_widget.reset_camera()

    def closeEvent(self, event):
        self.pv_widget.close()
        self.pv_widget.deleteLater()
        event.accept()