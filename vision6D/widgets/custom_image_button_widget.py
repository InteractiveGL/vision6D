from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
        
class PreviewButton(QtWidgets.QPushButton):
    def __init__(self, text='', image_path=None, parent=None):
        super().__init__(text, parent)
        self.image_path = image_path
        self.preview_label = None
        self.setMouseTracking(True)
        self.pixmap = QtGui.QPixmap(self.image_path) if self.image_path else None

    def enterEvent(self, event):
        if self.pixmap:
            window = self.window()
            window_width = window.width()
            window_height = window.height()
            percentage = 0.3
            preview_width = int(window_width * percentage)
            preview_height = int(window_height * percentage)

            # Get the original image size
            pixmap_width = self.pixmap.width()
            pixmap_height = self.pixmap.height()
            preview_width = min(preview_width, pixmap_width)
            preview_height = min(preview_height, pixmap_height)

            scaled_pixmap = self.pixmap.scaled(preview_width, preview_height, Qt.KeepAspectRatio)

            self.preview_label = QtWidgets.QLabel()
            self.preview_label.setPixmap(scaled_pixmap)
            self.preview_label.adjustSize()

            button_pos = self.mapToGlobal(QPoint(0, 0))
            preview_x = button_pos.x() + self.width()
            preview_y = button_pos.y()

            screen = QtWidgets.QApplication.primaryScreen()
            screen_geometry = screen.availableGeometry()
            if preview_x + scaled_pixmap.width() > screen_geometry.width():
                preview_x = button_pos.x() - scaled_pixmap.width()
            if preview_y + scaled_pixmap.height() > screen_geometry.height():
                preview_y = screen_geometry.height() - scaled_pixmap.height()
            self.preview_label.move(preview_x, preview_y)
            self.preview_label.setWindowFlags(Qt.ToolTip)
            self.preview_label.show()

        super().enterEvent(event)

    def leaveEvent(self, event):
        if self.preview_label:
            self.preview_label.close()
            self.preview_label = None # avoid the memory leak
        super().leaveEvent(event)

class CustomImageButtonWidget(QtWidgets.QWidget):
    colorChanged = pyqtSignal(str, str) 
    def __init__(self, button_name, image_path=None, parent=None):
        super(CustomImageButtonWidget, self).__init__(parent)
        self.setFixedHeight(30)
        self.image_path = image_path

        # Main layout for the widget
        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Container widget for the buttons
        button_container = QtWidgets.QWidget()
        button_layout = QtWidgets.QGridLayout(button_container)
        button_layout.setContentsMargins(0, 0, 5, 0)
        button_layout.setSpacing(0)

        # Create the main button using PreviewButton
        self.button = PreviewButton(button_name, image_path=self.image_path)
        self.button.setFixedHeight(30)
        button_layout.addWidget(self.button, 0, 0, 1, 1)

        # Add the button container to the main layout
        layout.addWidget(button_container)

        # Create the double spin box and add it to the layout
        self.double_spinbox = QtWidgets.QDoubleSpinBox()
        self.double_spinbox.setFixedHeight(28)
        self.double_spinbox.setMinimum(0.0)
        self.double_spinbox.setMaximum(1.0)
        self.double_spinbox.setDecimals(2)
        self.double_spinbox.setSingleStep(0.05)
        layout.addWidget(self.double_spinbox)

        # Set the stretch factors
        layout.setStretch(0, 20)
        layout.setStretch(1, 1)
        layout.setStretch(2, 1)

        # Set the layout for the widget
        self.setLayout(layout)