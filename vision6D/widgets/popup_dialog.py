# General import
import numpy as np

# Qt5 import
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

# self defined package import
np.set_printoptions(suppress=True)

class PopUpDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, on_button_click=None):
        super().__init__(parent)

        self.setWindowTitle("Vision6D - Colors")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint) # Disable the question mark

        button_grid = QtWidgets.QGridLayout()
        colors = ["nocs", "cyan", "magenta", "yellow", "lime", "latlon", "dodgerblue", "darkviolet", "darkorange", "forestgreen"]

        button_count = 0
        for i in range(2):
            for j in range(5):
                name = f"{colors[button_count]}"
                button = QtWidgets.QPushButton(name)
                button.clicked.connect(lambda _, idx=name: on_button_click(str(idx)))
                button_grid.addWidget(button, j, i)
                button_count += 1

        self.setLayout(button_grid)