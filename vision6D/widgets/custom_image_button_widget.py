from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QPoint, pyqtSignal

class PreviewButton(QtWidgets.QPushButton):
    active_preview_label = None  # Class variable to track the active preview label
    def __init__(self, text='', image_path=None, parent=None):
        super().__init__(text, parent)
        self.image_path = image_path
        self.preview_label = None
        self.is_closing = False  # Flag to indicate if the widget is closing
        self.setMouseTracking(True)
        self.pixmap = QtGui.QPixmap(self.image_path) if self.image_path else None

    def enterEvent(self, event):
        # Close any existing preview labels
        if PreviewButton.active_preview_label:
            PreviewButton.active_preview_label.close()
            PreviewButton.active_preview_label.deleteLater()
            PreviewButton.active_preview_label = None

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

            # Update the active preview label
            PreviewButton.active_preview_label = self.preview_label

        super().enterEvent(event)

    def leaveEvent(self, event):
        if self.is_closing: return  # Skip if the widget is closing
        if self.preview_label:
            self.preview_label.close()
            self.preview_label.deleteLater()
            # Reset the active preview label if it belongs to this button
            if PreviewButton.active_preview_label == self.preview_label:
                PreviewButton.active_preview_label = None
            self.preview_label = None  # Avoid memory leak
        super().leaveEvent(event)

    def closeEvent(self, event):
        self.is_closing = True  # Set the closing flag
        # Clean up resources
        if self.preview_label:
            self.preview_label.close()
            self.preview_label.deleteLater()
            self.preview_label = None
        if self.pixmap:
            self.pixmap = None
        super().closeEvent(event)

class SquareButton(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)

    def resizeEvent(self, event):
        self.setFixedSize(20, 20)
        
class CustomImageButtonWidget(QtWidgets.QWidget):
    mirrorXChanged = pyqtSignal(str)
    mirrorYChanged = pyqtSignal(str)
    def __init__(self, button_name, image_path=None, parent=None):
        super(CustomImageButtonWidget, self).__init__(parent)
        self.main_window = parent
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

        # Create the additional buttons
        self.mirror_x_button = SquareButton("|")
        self.mirror_y_button = SquareButton("—")
        self.mirror_x_button.clicked.connect(self.on_mirror_x_clicked)
        self.mirror_y_button.clicked.connect(self.on_mirror_y_clicked)

        # Create a horizontal layout for the square buttons and spacer
        square_button_layout = QtWidgets.QHBoxLayout()
        square_button_layout.setContentsMargins(0, 0, 0, 0)
        square_button_layout.setSpacing(5)  # Adjust spacing between buttons
        square_button_layout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # Add the square buttons to the layout
        square_button_layout.addWidget(self.mirror_x_button)
        square_button_layout.addWidget(self.mirror_y_button)

        # Optionally, add spacing to the right
        square_button_layout.addSpacing(5)

        # Add the square button layout to the main button layout
        button_layout.addLayout(square_button_layout, 0, 0, 1, 1, Qt.AlignRight | Qt.AlignVCenter)

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
    
    def contextMenuEvent(self, event):
        context_menu = QtWidgets.QMenu(self)
        remove_action = context_menu.addAction("Remove")
        remove_action.triggered.connect(self.remove_self)
        context_menu.exec_(event.globalPos())

    def remove_self(self):
        self.main_window.remove_image_button(self.button)

    def on_mirror_x_clicked(self):
        self.mirrorXChanged.emit("x")

    def on_mirror_y_clicked(self):
        self.mirrorYChanged.emit("y")