import numpy as np
import matplotlib
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
        self.original_pixmap = pixmap
        self.scaled_pixmap = pixmap
        self.setPixmap(self.scaled_pixmap)
        self.setContentsMargins(0, 0, 0, 0)
        self.points = []
        self.scale_factor_x = 1.0
        self.scale_factor_y = 1.0

    def resizeEvent(self, event):
        self.scaled_pixmap = self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self.scaled_pixmap)
        self.scale_factor_x = self.original_pixmap.width() / self.scaled_pixmap.width()
        self.scale_factor_y = self.original_pixmap.height() / self.scaled_pixmap.height()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Calculate offset
            label_width, label_height = self.width(), self.height()
            pixmap_width, pixmap_height = self.scaled_pixmap.width(), self.scaled_pixmap.height()
            self.offset_x = (label_width - pixmap_width) / 2
            self.offset_y = (label_height - pixmap_height) / 2
            x = event.pos().x() - self.offset_x
            y = event.pos().y() - self.offset_y
            # Check if the click is within the image area
            if 0 <= x < pixmap_width and 0 <= y < pixmap_height:
                # Map to original image coordinates
                orig_x = x * self.scale_factor_x
                orig_y = y * self.scale_factor_y
                self.points.append((orig_x, orig_y))
                self.update()
        elif event.button() == Qt.RightButton and self.points:
            self.points.pop()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QColor(255, 0, 0))
        font = QtGui.QFont()
        font.setPointSize(18)  # Set font size to 12 (you can change this to any size)
        painter.setFont(font)
        for i, (orig_x, orig_y) in enumerate(self.points):
            # Map original coordinates to scaled image coordinates
            x = orig_x / self.scale_factor_x + self.offset_x
            y = orig_y / self.scale_factor_y + self.offset_y
            point = QtCore.QPointF(x, y)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)))
            painter.drawEllipse(point, 8, 8)
            text_point = QtCore.QPoint(int(point.x() + 12), int(point.y() + 12))
            painter.drawText(text_point, str(i + 1))

    def get_2d_points(self):
        return np.array(self.points).astype('int32')

class PnPWindow(QtWidgets.QWidget):
    transformation_matrix_computed = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, image_source, mesh_model, camera_intrinsics):
        super().__init__()

        self.setWindowTitle("2D-to-3D Registration Using the PnP Algorithm") 

        self.mesh_model = mesh_model
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
        self.plot_3d_model()
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.pv_widget)
        right_panel.setLayout(right_layout)
        content_layout.addWidget(right_panel)

        # Set stretch factors to make 2D and 3D views half and half
        content_layout.setStretch(0, 1)
        content_layout.setStretch(1, 1)

        # Add the content (2D image + 3D view) to the main vertical layout
        layout.addLayout(content_layout)

        # Add the submit button at the bottom
        submit_button = QtWidgets.QPushButton("(At least four points to perform PnP registration) Submit")
        submit_button.clicked.connect(self.submit_to_pnp_register)
        layout.addWidget(submit_button)

        # Show the window maximized, and still include the close button
        self.showMaximized()

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
                flags=cv2.SOLVEPNP_EPNP)
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
        label_actor = self.pv_widget.add_point_labels([point], [label_text], show_points=False, font_size=20, text_color="red", shape='rect', shape_opacity=1, pickable=False, reset_camera=False)
        self.point_labels.append(label_actor)

    def right_click_callback(self, *args):
        if self.picked_3d_points:
            self.picked_3d_points.pop()
            self.pv_widget.renderer.RemoveActor(self.point_labels.pop())
            self.pv_widget.render()

    def plot_3d_model(self):
        scalars = None
        if self.mesh_model.color == "nocs": scalars = utils.color_mesh_nocs(self.mesh_model.pv_obj.points)
        elif self.mesh_model.color == "texture": scalars = np.load(self.mesh_model.texture_path) / 255
        if scalars is not None: 
            self.mesh_actor = self.pv_widget.add_mesh(self.mesh_model.pv_obj, scalars=scalars, rgb=True, opacity=1, show_scalar_bar=False)
        else: 
            self.mesh_actor = self.pv_widget.add_mesh(self.mesh_model.pv_obj, opacity=1, show_scalar_bar=False)
            self.mesh_actor.GetMapper().SetScalarVisibility(0)
            self.mesh_actor.GetProperty().SetColor(matplotlib.colors.to_rgb(self.mesh_model.color))
        
        self.pv_widget.enable_surface_point_picking(callback=self.point_picking_callback, show_message=False, show_point=False, left_clicking=True, color='red')
        self.pv_widget.iren.add_observer("RightButtonPressEvent", self.right_click_callback)
        self.pv_widget.reset_camera()

    def closeEvent(self, event):
        self.pv_widget.close()
        self.pv_widget.deleteLater()
        event.accept()