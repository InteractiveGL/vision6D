'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: camera_props_input_dialog.py
@time: 2023-07-03 20:30
@desc: pop window for camera props input dialog
'''

# General import
import numpy as np

# Qt5 import
from PyQt5 import QtWidgets

# self defined package import
np.set_printoptions(suppress=True)

class CameraPropsInputDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, 
                    line1=(None, None), 
                    line2=(None, None), 
                    line3=(None, None), 
                    line4=(None, None), 
                    line5=(None, None),
                    line6=(None, None),
                    line7=(None, None)):
        super().__init__(parent)

        self.args1 = QtWidgets.QLineEdit(self, text=str(line1[1]))
        self.args2 = QtWidgets.QLineEdit(self, text=str(line2[1]))
        self.args3 = QtWidgets.QLineEdit(self, text=str(line3[1]))
        self.args4 = QtWidgets.QLineEdit(self, text=str(line4[1]))
        self.args5 = QtWidgets.QLineEdit(self, text=str(line5[1]))
        self.args6 = QtWidgets.QLineEdit(self, text=str(line6[1]))
        self.args7 = QtWidgets.QLineEdit(self, text=str(line7[1]))

        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)

        layout = QtWidgets.QFormLayout(self)
        layout.addRow(f"{line1[0]}", self.args1)
        layout.addRow(f"{line2[0]}", self.args2)
        layout.addRow(f"{line3[0]}", self.args3)
        layout.addRow(f"{line4[0]}", self.args4)
        layout.addRow(f"{line5[0]}", self.args5)
        layout.addRow(f"{line6[0]}", self.args6)
        layout.addRow(f"{line7[0]}", self.args7)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.args1.text(), 
                self.args2.text(), 
                self.args3.text(),
                self.args4.text(),
                self.args5.text(),
                self.args6.text(),
                self.args7.text())