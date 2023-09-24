'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: get_pose_dialog.py
@time: 2023-07-03 20:31
@desc: get the text dialog for user input
'''

# General import
import numpy as np

# Qt5 import
from PyQt5 import QtWidgets

# self defined package import
np.set_printoptions(suppress=True)

class GetPoseDialog(QtWidgets.QDialog):
    def __init__(self, pose, parent=None):
        super(GetPoseDialog, self).__init__(parent)
        self.pose = pose
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
        text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                self.pose[0, 0], self.pose[0, 1], self.pose[0, 2], self.pose[0, 3], 
                self.pose[1, 0], self.pose[1, 1], self.pose[1, 2], self.pose[1, 3], 
                self.pose[2, 0], self.pose[2, 1], self.pose[2, 2], self.pose[2, 3],
                self.pose[3, 0], self.pose[3, 1], self.pose[3, 2], self.pose[3, 3])
        self.textEdit.setPlainText(text)
        self.btnSubmit = QtWidgets.QPushButton("Submit", self)
        self.btnSubmit.clicked.connect(self.submit_text)
        layout.addWidget(self.hboxWidget)
        layout.addWidget(self.textEdit)
        layout.addWidget(self.btnSubmit)
        
    def submit_text(self):
        self.accept()

    def load_from_file(self):
        pose_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.npy)")
        if pose_path:
            gt_pose = np.load(pose_path)
            text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                gt_pose[0, 0], gt_pose[0, 1], gt_pose[0, 2], gt_pose[0, 3], 
                gt_pose[1, 0], gt_pose[1, 1], gt_pose[1, 2], gt_pose[1, 3], 
                gt_pose[2, 0], gt_pose[2, 1], gt_pose[2, 2], gt_pose[2, 3],
                gt_pose[3, 0], gt_pose[3, 1], gt_pose[3, 2], gt_pose[3, 3])
            self.textEdit.setPlainText(text)

    def get_text(self):
        return self.textEdit.toPlainText()