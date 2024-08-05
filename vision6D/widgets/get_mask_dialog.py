'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: get_mask_dialog.py
@time: 2023-07-03 20:31
@desc: get the text dialog for user input
'''

# General import
import numpy as np

# Qt5 import
from PyQt5 import QtWidgets

# self defined package import
np.set_printoptions(suppress=True)

class GetMaskDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(GetMaskDialog, self).__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.mask_path = None
        self.setWindowTitle("Vision6D")
        self.introLabel = QtWidgets.QLabel("Input the Mask:")
        self.btnloadfromfile = QtWidgets.QPushButton("Load from file", self)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.introLabel)
        hbox.addWidget(self.btnloadfromfile)
        hbox.setContentsMargins(0, 0, 0, 0)
        self.hboxWidget = QtWidgets.QWidget()
        self.hboxWidget.setLayout(hbox)
        self.btnloadfromfile.clicked.connect(self.load_from_file)
        self.textEdit = QtWidgets.QTextEdit(self)
        self.btnSubmit = QtWidgets.QPushButton("Submit", self)
        self.btnSubmit.clicked.connect(self.submit_text)
        layout.addWidget(self.hboxWidget)
        layout.addWidget(self.textEdit)
        layout.addWidget(self.btnSubmit)
        
    def submit_text(self):
        self.accept()

    def load_from_file(self):
        mask_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.npy *.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)") 
        if mask_path:
            self.mask_path = mask_path
            if ".npy" in self.mask_path:
                self.textEdit.setPlainText(f"{np.load(self.mask_path).squeeze()}")
            else:
                self.textEdit.setPlainText(f"{self.mask_path}")

    def get_text(self):
        return self.textEdit.toPlainText()