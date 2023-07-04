'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: video_sampler.py
@time: 2023-07-03 20:33
@desc: create the video sampler
'''

# General import
import numpy as np
import cv2
import pathlib

# Qt5 import
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt

# self defined package import
np.set_printoptions(suppress=True)

class VideoSampler(QtWidgets.QDialog):
    def __init__(self, video_player, fps, parent=None):
        super(VideoSampler, self).__init__(parent)
        self.setWindowTitle("Vision6D - Video Sampler")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint) # Disable the question mark
        self.setFixedSize(800, 500)
        
        self.video_player = video_player
        self.fps = fps

        layout = QtWidgets.QVBoxLayout(self)

        # Create QLabel for the top
        label1 = QtWidgets.QLabel("How often should we sample this video?", self)
        font = QtGui.QFont('Times', 14)
        font.setBold(True)
        label1.setFont(font)

        label1.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label1)

        line = QtWidgets.QFrame(self)
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet("color: grey")
        layout.addWidget(line)

        # Load the video
        self.video_path = self.video_player.video_path
        self.cap = self.video_player.cap
        self.frame_count = self.video_player.frame_count
        video_width = self.video_player.video_width
        video_height = self.video_player.video_height

        if video_width > 600 and video_height > 400: self.video_size = int(video_width // 4), int(video_height // 4)
        else: self.video_size = video_width, video_height
        
        # Create a QLabel to hold the thumbnail
        self.thumbnail_label = QtWidgets.QLabel(self)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret: thumbnail_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Load the image using QPixmap
        img = QtGui.QImage(thumbnail_frame.tobytes(), thumbnail_frame.shape[1], thumbnail_frame.shape[0], thumbnail_frame.shape[2]*thumbnail_frame.shape[1], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(img)
        
        # Resize the QPixmap to the desired thumbnail size
        thumbnail = pixmap.scaled(*self.video_size)  # Change the size to fit your needs
        
        # Set the QPixmap as the image displayed by the QLabel
        self.thumbnail_label.setPixmap(thumbnail)
        
        layout.addWidget(self.thumbnail_label, alignment=Qt.AlignCenter)

        # Calculate and print the video duration in seconds
        total_seconds = self.frame_count / self.fps
        minutes, remainder_seconds = divmod(total_seconds, 60)

        # Create QLabel for the bottom
        label2 = QtWidgets.QLabel(f"{pathlib.Path(self.video_path).stem}{pathlib.Path(self.video_path).suffix} ({int(minutes)}m{int(remainder_seconds)}s)", self)
        font = QtGui.QFont('Times', 10)
        font.setBold(True)
        label2.setFont(font)
        label2.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label2, alignment=Qt.AlignCenter)

        hlayout = QtWidgets.QGridLayout()
        self.label_frame_rate = QtWidgets.QLabel(f"Frame per step: ")
        self.label_frame_rate.setContentsMargins(80, 0, 0, 0)
        self.label_frame_rate.setFont(font)
        hlayout.addWidget(self.label_frame_rate, 0, 0)
        self.step_spinbox = QtWidgets.QSpinBox()
        self.step_spinbox.setMinimum(1)
        self.step_spinbox.setMaximum(self.frame_count)
        self.step_spinbox.setValue(self.fps)
        self.step_spinbox.valueChanged.connect(self.step_spinbox_value_changed)
        hlayout.addWidget(self.step_spinbox, 0, 1)
        self.output_size_label = QtWidgets.QLabel(f"Total output images: ")
        self.output_size_label.setContentsMargins(80, 0, 0, 0)
        self.output_size_label.setFont(font)
        hlayout.addWidget(self.output_size_label, 0, 2)
        self.output_spinbox = QtWidgets.QSpinBox()
        self.output_spinbox.setMinimum(1)
        self.output_spinbox.setMaximum(self.frame_count)
        self.output_spinbox.setValue(round(self.frame_count // self.fps))
        self.output_spinbox.valueChanged.connect(self.output_spinbox_value_changed)
        hlayout.addWidget(self.output_spinbox, 0, 3)
        layout.addLayout(hlayout)

        line = QtWidgets.QFrame(self)
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet("color: grey")
        layout.addWidget(line)

        accept_button = QtWidgets.QPushButton('Choose Frame Rate', self)
        font = QtGui.QFont('Times', 12); font.setBold(True)
        accept_button.setFont(font)
        accept_button.setFixedSize(300, 40)
        accept_button.clicked.connect(self.accept)
        layout.addWidget(accept_button, alignment=Qt.AlignRight)

        self.setLayout(layout)

    def step_spinbox_value_changed(self, value):
        self.fps = value
        self.output_spinbox.setValue(round(self.frame_count // self.fps))
        
    def output_spinbox_value_changed(self, value):
        self.fps = round(self.frame_count // value)
        self.step_spinbox.setValue(self.fps)

    def closeEvent(self, event):
        event.ignore()
        super().closeEvent(event)

    def accept(self):
        super().accept()