
import numpy as np
import cv2

from ..widgets import VideoSampler, VideoPlayer
from .singleton import Singleton

class VideoStore(metaclass=Singleton):

    def __init__(self):
        self.reset()
        
    def reset(self):
        self.video_path = None
        self.current_frame = 0

    def add_video(self, video_path):
        self.video_path = video_path
        self.video_player = VideoPlayer(self.video_path, self.current_frame)
        self.fps = round(self.video_player.fps)
        self.total_frame = self.video_player.frame_count
        
    def load_per_frame_info(self):
        self.video_player.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.video_player.cap.read()
        if ret: 
            video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return video_frame

    def sample_video(self):
        return VideoSampler(self.video_player, self.fps)
    
    def prev_frame(self):
        current_frame = self.current_frame - self.fps
        np.clip(current_frame, 0, self.total_frame)
        self.current_frame = current_frame
        return current_frame
        
    def next_frame(self):
        current_frame = self.current_frame + self.fps
        np.clip(current_frame, 0, self.total_frame)
        self.current_frame = current_frame
        return current_frame
                