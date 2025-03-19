from PyQt5 import QtWidgets

class CameraControlWidget(QtWidgets.QWidget):
    def __init__(self, label, unit, axis_range, parent=None):
        super().__init__(parent)
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QtWidgets.QLabel(label)
        layout.addWidget(self.label)
        
        self.spin_box = QtWidgets.QDoubleSpinBox()
        self.spin_box.setRange(-axis_range, axis_range)
        self.spin_box.setMinimumHeight(25)
        self.spin_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        layout.addWidget(self.spin_box)

        self.unit = QtWidgets.QLabel(unit)
        layout.addWidget(self.unit)


