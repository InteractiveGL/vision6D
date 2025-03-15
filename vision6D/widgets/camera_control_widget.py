from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

class CameraControlWidget(QtWidgets.QWidget):
    def __init__(self, axis_label, axis_unit, parent=None):
        super().__init__(parent)
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Axis label
        self.axis_label = QtWidgets.QLabel(axis_label)
        layout.addWidget(self.axis_label)
        
        # Spin box
        self.spin_box = QtWidgets.QSpinBox()
        self.spin_box.setRange(-100, 100)
        layout.addWidget(self.spin_box)

        # Axis unit label
        self.axis_unit = QtWidgets.QLabel(axis_unit)
        layout.addWidget(self.axis_unit)

        self.spin_box.valueChanged.connect(self.on_value_changed)

    def on_value_changed(self, new_value):
        print(f"{self.axis_label.text()} changed to {new_value}")



