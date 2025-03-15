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
        self.mirror_x_button = QtWidgets.QPushButton("|", self)
        self.mirror_y_button = QtWidgets.QPushButton("—", self)

        # Horizontal Layout
        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)

        # Add Widgets to Horizontal Layout
        hbox.addWidget(self.introLabel)
        hbox.addStretch()  # Add stretch to push buttons to the right
        hbox.addWidget(self.btnloadfromfile)
        hbox.addWidget(self.mirror_x_button)
        hbox.addWidget(self.mirror_y_button)

        # Create a Widget to Hold the Horizontal Layout
        self.hboxWidget = QtWidgets.QWidget()
        self.hboxWidget.setLayout(hbox)

        # Connect Buttons to Their Slot Methods
        self.btnloadfromfile.clicked.connect(self.load_from_file)
        self.mirror_x_button.clicked.connect(self.on_mirror_x_clicked)
        self.mirror_y_button.clicked.connect(self.on_mirror_y_clicked)

        # Text Edit Field
        self.textEdit = QtWidgets.QTextEdit(self)
        self.update_text_edit()
        self.btnSubmit = QtWidgets.QPushButton("Submit", self)
        self.btnSubmit.clicked.connect(self.submit_text)
        layout.addWidget(self.hboxWidget)
        layout.addWidget(self.textEdit)
        layout.addWidget(self.btnSubmit)

    def on_mirror_x_clicked(self):
        self.pose = self.mirror_pose(self.pose, 'x')
        self.update_text_edit()

    def on_mirror_y_clicked(self):
        self.pose = self.mirror_pose(self.pose, 'y')
        self.update_text_edit()

    def mirror_pose(self, transformation_matrix, direction):
        if direction == 'x': mirror_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        elif direction == 'y': mirror_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return mirror_matrix @ transformation_matrix

    def update_text_edit(self):
        text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                self.pose[0, 0], self.pose[0, 1], self.pose[0, 2], self.pose[0, 3], 
                self.pose[1, 0], self.pose[1, 1], self.pose[1, 2], self.pose[1, 3], 
                self.pose[2, 0], self.pose[2, 1], self.pose[2, 2], self.pose[2, 3],
                self.pose[3, 0], self.pose[3, 1], self.pose[3, 2], self.pose[3, 3])
        self.textEdit.setPlainText(text)
        
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
    
    def get_pose(self):
        return self.pose