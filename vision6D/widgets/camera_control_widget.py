from PyQt5 import QtWidgets

class CameraControlWidget(QtWidgets.QWidget):
    def __init__(self, axis_label, axis_unit, axis_range, parent=None):
        super().__init__(parent)
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Axis label
        self.axis_label = QtWidgets.QLabel(axis_label)
        layout.addWidget(self.axis_label)
        
        # Spin box
        self.spin_box = QtWidgets.QDoubleSpinBox()
        self.spin_box.setRange(-axis_range, axis_range)
        self.spin_box.setMinimumHeight(25)
        self.spin_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        layout.addWidget(self.spin_box)

        # Axis unit label
        self.axis_unit = QtWidgets.QLabel(axis_unit)
        layout.addWidget(self.axis_unit)


