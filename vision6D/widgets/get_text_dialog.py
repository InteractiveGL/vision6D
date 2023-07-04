'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: get_text_dialog.py
@time: 2023-07-03 20:31
@desc: get the text dialog for user input
'''

# General import
import numpy as np

# Qt5 import
from PyQt5 import QtWidgets

# self defined package import
np.set_printoptions(suppress=True)

class GetTextDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(GetTextDialog, self).__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        
        self.setWindowTitle("Vision6D")
        self.introLabel = QtWidgets.QLabel("Input the Ground Truth Pose:")
        self.btnloadfromfile = QtWidgets.QPushButton("Load from file", self)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.introLabel)
        hbox.addWidget(self.btnloadfromfile)
        hbox.setContentsMargins(0, 0, 0, 0)
        self.hboxWidget = QtWidgets.QWidget()
        self.hboxWidget.setLayout(hbox)

        self.btnloadfromfile.clicked.connect(self.load_from_file)
        self.textEdit = QtWidgets.QTextEdit(self)
        self.textEdit.setPlainText(f"[[1, 0, 0, 0], \n[0, 1, 0, 0], \n[0, 0, 1, 0], \n[0, 0, 0, 1]]")
        self.btnSubmit = QtWidgets.QPushButton("Submit", self)
        self.btnSubmit.clicked.connect(self.submit_text)

        layout.addWidget(self.hboxWidget)
        layout.addWidget(self.textEdit)
        layout.addWidget(self.btnSubmit)

    def submit_text(self):
        self.user_text = self.textEdit.toPlainText()
        self.accept()

    def load_from_file(self):
        pose_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.npy)")
        if pose_path:
            gt_pose = np.load(pose_path)
            self.textEdit.setPlainText(f"[[{np.around(gt_pose[0, 0], 8)}, {np.around(gt_pose[0, 1], 8)}, {np.around(gt_pose[0, 2], 8)}, {np.around(gt_pose[0, 3], 8)}],\n"
                                    f"[{np.around(gt_pose[1, 0], 8)}, {np.around(gt_pose[1, 1], 8)}, {np.around(gt_pose[1, 2], 8)}, {np.around(gt_pose[1, 3], 8)}],\n"
                                    f"[{np.around(gt_pose[2, 0], 8)}, {np.around(gt_pose[2, 1], 8)}, {np.around(gt_pose[2, 2], 8)}, {np.around(gt_pose[2, 3], 8)}],\n"
                                    f"[{np.around(gt_pose[3, 0], 8)}, {np.around(gt_pose[3, 1], 8)}, {np.around(gt_pose[3, 2], 8)}, {np.around(gt_pose[3, 3], 8)}]]")

    def get_text(self):
        return self.user_text