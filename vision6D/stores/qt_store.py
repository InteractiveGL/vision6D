import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from .singleton import Singleton
from ..stores import PlotStore

np.set_printoptions(suppress=True)

class QtStore(metaclass=Singleton):
    def __init__(self):
        super().__init__()

        self.plot_store = PlotStore()

        self.track_actors_names = []

        # Create the color dropdown menu
        self.color_button = QtWidgets.QPushButton("Color")

        self.hintLabel = QtWidgets.QLabel(self.plot_store.plotter)
        self.hintLabel.setText("Drag and drop a file here...")
        self.hintLabel.setStyleSheet("""
                                    color: white; 
                                    background-color: rgba(0, 0, 0, 127); 
                                    padding: 10px;
                                    border: 2px dashed gray;
                                    """)
        self.hintLabel.setAlignment(Qt.AlignCenter)

        self.output_text = QtWidgets.QTextEdit()
        self.output_text.setReadOnly(True)