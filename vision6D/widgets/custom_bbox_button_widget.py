from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from . import PopUpDialog

class SquareButton(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)

    def resizeEvent(self, event):
        self.setFixedSize(40, 40)

class CustomBboxButtonWidget(QtWidgets.QWidget):
    colorChanged = pyqtSignal(str, str) 
    def __init__(self, button_name, parent=None):
        super(CustomBboxButtonWidget, self).__init__(parent)
        self.setFixedHeight(50)

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
        self.button = QtWidgets.QPushButton(button_name)
        self.button.setFixedHeight(50)
        button_layout.addWidget(self.button, 0, 0, 1, 1)

        # Create the square button
        self.square_button = SquareButton()
        self.square_button.setFixedSize(35, 35)
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

    def contextMenuEvent(self, event):
        context_menu = QtWidgets.QMenu(self)
        remove_action = context_menu.addAction("Remove")
        remove_action.triggered.connect(self.remove_self)
        context_menu.exec_(event.globalPos())

    def remove_self(self):
        self.parent().remove_custom_button_widget(self.button)

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
        popup = PopUpDialog(self, on_button_click=lambda text: self.update_square_button_color(text, popup))
        button_position = self.square_button.mapToGlobal(QPoint(0, 0))
        popup.move(button_position + QPoint(self.square_button.width(), 0))
        popup.exec_()

