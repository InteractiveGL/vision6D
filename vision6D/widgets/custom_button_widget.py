from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from . import PopUpDialog

class SquareButton(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)

    def resizeEvent(self, event):
        self.setFixedSize(40, 40)
        
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

class CustomButtonWidget(QtWidgets.QWidget):
    colorChanged = pyqtSignal(str, str) 
    def __init__(self, button_name, image_path=None, parent=None):
        super(CustomButtonWidget, self).__init__(parent)
        self.setFixedHeight(50)
        self.image_path = image_path

        # Main layout for the widget
        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Container widget for the buttons
        button_container = QtWidgets.QWidget()
        button_layout = QtWidgets.QGridLayout(button_container)
        button_layout.setContentsMargins(0, 0, 10, 0)
        button_layout.setSpacing(0)

        # Create the main button using PreviewButton
        self.button = PreviewButton(button_name, image_path=self.image_path)
        self.button.setFixedHeight(50)
        button_layout.addWidget(self.button, 0, 0, 1, 1)

        # Create the square button
        self.square_button = SquareButton()
        self.square_button.setFixedSize(35, 35)
        if self.image_path is not None:
            pixmap = QtGui.QPixmap(self.image_path)
            scaled_pixmap = pixmap.scaled(self.square_button.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.square_button.setIcon(QtGui.QIcon(scaled_pixmap))
            self.square_button.setIconSize(self.square_button.size())
            self.square_button.clicked.connect(self.show_image_preview)
        else: 
            self.square_button.clicked.connect(self.show_color_popup)

        # Create a horizontal layout for the square button and spacer
        square_button_layout = QtWidgets.QHBoxLayout()
        square_button_layout.addWidget(self.square_button)
        square_button_layout.addSpacing(5)  # Add 10 pixels of space to the right
        square_button_layout.setContentsMargins(0, 0, 0, 0)
        square_button_layout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # Add the square button layout to the main button layout
        button_layout.addLayout(square_button_layout, 0, 0, 1, 1, Qt.AlignRight | Qt.AlignVCenter)

        # Add the button container to the main layout
        layout.addWidget(button_container)

        # Create the double spin box and add it to the layout
        self.double_spinbox = QtWidgets.QDoubleSpinBox()
        self.double_spinbox.setFixedHeight(45)
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

    def update_square_button_color(self, text, popup):
        self.square_button.setObjectName(text)
        if text == 'nocs' or text == 'texture':
            gradient_str = """
            background-color: qlineargradient(
                spread:pad, x1:0, y1:0, x2:1, y2:1,
                stop:0 red, stop:0.17 orange, stop:0.33 yellow,
                stop:0.5 green, stop:0.67 blue, stop:0.83 indigo, stop:1 violet);
            """
            self.square_button.setStyleSheet(gradient_str)
        else:
            self.square_button.setStyleSheet(f"background-color: {text}")
        self.colorChanged.emit(text, self.button.text()) # the order is important (color, name)
        popup.close() # automatically close the popup window

    def show_color_popup(self):
        button_name = self.button.text()
        if button_name != 'image':
            popup = PopUpDialog(self, on_button_click=lambda text: self.update_square_button_color(text, popup))
            button_position = self.square_button.mapToGlobal(QPoint(0, 0))
            popup.move(button_position + QPoint(self.square_button.width(), 0))
            popup.exec_()

    def show_image_preview(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Image Preview")
        dialog.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint | Qt.WindowCloseButtonHint | Qt.CustomizeWindowHint)
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(self.image_path)
        window = self.window()
        preview_width = int(window.width() * 0.8)
        preview_height = int(window.height() * 0.8)
        scaled_pixmap = pixmap.scaled(preview_width, preview_height, Qt.KeepAspectRatio)
        label.setPixmap(scaled_pixmap)
        layout.addWidget(label)
        dialog.exec_()