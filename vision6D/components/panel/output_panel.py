from PyQt5 import QtWidgets, QtGui

from ...stores import QtStore

class OutputPanel:
    def __init__(self, output):

        # Save reference
        self.output = output

        # Add a spacer to the top of the main layout
        output_layout = QtWidgets.QVBoxLayout()
        output_layout.setContentsMargins(10, 15, 10, 0)

        #* Create the top widgets (layout)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Create Grid layout for function buttons
        grid_layout = QtWidgets.QGridLayout()

        # Create the set camera button
        copy_text_button = QtWidgets.QPushButton("Copy")
        copy_text_button.clicked.connect(self.copy_output_text)
        grid_layout.addWidget(copy_text_button, 0, 2, 1, 1)

        # Create the actor pose button
        clear_text_button = QtWidgets.QPushButton("Clear")
        clear_text_button.clicked.connect(self.clear_output_text)
        grid_layout.addWidget(clear_text_button, 0, 3, 1, 1)

        grid_widget = QtWidgets.QWidget()
        grid_widget.setLayout(grid_layout)
        top_layout.addWidget(grid_widget)
        output_layout.addLayout(top_layout)

        # Access to the system clipboard
        self.qt_store = QtStore()
        output_layout.addWidget(self.qt_store.output_text)
        self.output.setLayout(output_layout)


    def copy_output_text(self):
        self.qt_store.clipboard.setText(self.qt_store.output_text.toPlainText())
        
    def clear_output_text(self):
        self.qt_store.output_text.clear()
