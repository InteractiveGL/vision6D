'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: video_player.py
@time: 2023-07-03 20:33
@desc: create the video player
'''

# General import
import numpy as np
import cv2

# Qt5 import
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt

# self defined package import
np.set_printoptions(suppress=True)

class VideoPlayer(QtWidgets.QDialog):
    def __init__(self, video_path, current_frame):
        super().__init__()

        self.setWindowTitle("Vision6D - Video Player")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint) # Disable the question mark
        self.setFixedSize(1000, 600)

        # Add shortcut to the right, left, space buttons
        QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self).activated.connect(self.next_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self).activated.connect(self.prev_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self).activated.connect(self.play_pause_video)

        self.video_path = video_path
        self.play = False

        # Load the video
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.video_width > 960 and self.video_height > 540: self.video_size = int(self.video_width // 2), int(self.video_height // 2)
        else: self.video_size = self.video_width, self.video_height
        
        self.current_frame = current_frame

        self.playback_speeds = [0.1, 0.2, 0.5, 1.0, 4.0, 16.0]  # different speeds
        self.current_playback_speed = 1  # Default speed is 1

        self.play_pause_button = QtWidgets.QToolButton(self)
        self.play_pause_button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.play_pause_button.setText(f'Play/Pause ({self.current_frame}/{self.frame_count})')
        self.play_pause_button.clicked.connect(self.play_pause_video)
        
        # create the overall layout
        self.layout = QtWidgets.QVBoxLayout(self)
        # create the button layout
        self.button_layout = QtWidgets.QHBoxLayout()

        self.label = QtWidgets.QLabel(self)
        self.layout.addWidget(self.label, 0, QtCore.Qt.AlignCenter)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_moved)
        self.slider.setMaximum(self.frame_count - 1)
        self.layout.addWidget(self.slider)

        self.play_pause_menu = QtWidgets.QMenu(self)
        self.play_action = QtWidgets.QAction('Play', self, triggered=self.play_video)
        self.pause_action = QtWidgets.QAction('Pause', self, triggered=self.pause_video)
        self.speed_menu = QtWidgets.QMenu('Playback Speed', self)

        self.speed_action_group = QtWidgets.QActionGroup(self.speed_menu)
        self.speed_action_group.setExclusive(True)
        for speed in self.playback_speeds:
            speed_action = QtWidgets.QAction(f'{speed}x', self.speed_menu, checkable=True)
            speed_action.triggered.connect(lambda _, s=speed: self.change_speed(speed=s))
            if speed == self.current_playback_speed: speed_action.setChecked(True)
            self.speed_menu.addAction(speed_action)
            self.speed_action_group.addAction(speed_action)
            
        self.prev_button = QtWidgets.QPushButton('Previous Frame', self)
        self.prev_button.clicked.connect(self.prev_frame)
        self.prev_button.setFixedSize(300, 30)
        self.button_layout.addWidget(self.prev_button)

        self.play_pause_menu.addActions([self.play_action, self.pause_action])
        self.play_pause_menu.addMenu(self.speed_menu)
        self.play_pause_button.setMenu(self.play_pause_menu)
        self.play_pause_button.setFixedSize(300, 30)
        self.button_layout.addWidget(self.play_pause_button)

        self.next_button = QtWidgets.QPushButton('Next Frame', self)
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setFixedSize(300, 30)
        self.button_layout.addWidget(self.next_button)

        self.layout.addLayout(self.button_layout)

        line = QtWidgets.QFrame(self)
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet("color: grey")
        self.layout.addWidget(line)

        accept_button = QtWidgets.QPushButton('Set selected frame', self)
        font = QtGui.QFont('Times', 12); font.setBold(True)
        accept_button.setFont(font)
        accept_button.setFixedSize(400, 40)
        accept_button.clicked.connect(self.accept)
        self.layout.addWidget(accept_button, alignment=Qt.AlignRight)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.isPlaying = False

        # Display frame
        self.update_frame()

    def slider_moved(self, value):
        self.current_frame = value
        self.update_frame()

    def change_speed(self, speed):
        self.current_playback_speed = speed
        self.play_video()

    def play_video(self):
        self.isPlaying = True
        self.timer.start(self.fps / self.current_playback_speed)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

    def pause_video(self):
        self.isPlaying = False
        self.timer.stop()
        self.prev_button.setEnabled(True)
        self.next_button.setEnabled(True)

    def play_pause_video(self):
        if self.isPlaying: self.pause_video()
        else: self.play_video()

    def next_frame(self):
        current_frame = self.current_frame + 1
        if current_frame <= self.frame_count: self.current_frame = current_frame
        self.slider.setValue(self.current_frame)

    def prev_frame(self):
        current_frame = self.current_frame - 1
        if current_frame >= 0: self.current_frame = current_frame
        self.slider.setValue(self.current_frame)

    def update_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.play_pause_button.setText(f'Play/Pause ({self.current_frame}/{self.frame_count})')
        ret, frame = self.cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QtGui.QImage(rgb_image.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            p = convert_to_qt_format.scaled(*self.video_size, QtCore.Qt.KeepAspectRatio)
            self.label.setPixmap(QtGui.QPixmap.fromImage(p))

    def closeEvent(self, event):
        event.ignore()
        super().closeEvent(event)

    def accept(self):
        super().accept()