'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: video_store.py
@time: 2023-07-03 20:25
@desc: create store for video related functions
'''

import cv2
import numpy as np

from . import Singleton
from ..widgets import VideoPlayer
from ..widgets import VideoSampler

from PyQt5 import QtWidgets

class VideoStore(metaclass=Singleton):
    def __init__(self):
        self.reset()

    def reset(self):
        self.video_path = None
        self.current_frame = 0
        self.video_player = None
        self.video_sampler = None

    def add_video(self, video_path):
        self.video_path = video_path
        try:
            self.video_player = VideoPlayer(self.video_path, self.current_frame)
            self.total_frame = self.video_player.frame_count
            self.fps = round(self.video_player.fps)
        except:
            self.reset()

    def play_video(self):
        res = self.video_player.exec_()
        if res == QtWidgets.QDialog.Accepted:
            self.current_frame = self.video_player.current_frame

    def sample_video(self):
        self.video_sampler = VideoSampler(self.video_player, self.fps)
        res = self.video_sampler.exec_()
        if res == QtWidgets.QDialog.Accepted: self.fps = round(self.video_sampler.fps)

    def load_per_frame_info(self):
        self.video_player.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.video_player.cap.read()
        if ret: 
            video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return video_frame
        else:
            return None
        
    def prev_frame(self):
        self.current_frame = self.current_frame - self.fps
        self.current_frame = np.clip(self.current_frame, 0, self.total_frame)
        self.video_player.slider.setValue(self.current_frame)
        
    def next_frame(self):
        self.current_frame = self.current_frame + self.fps
        self.current_frame = np.clip(self.current_frame, 0, self.total_frame)
        self.video_player.slider.setValue(self.current_frame)