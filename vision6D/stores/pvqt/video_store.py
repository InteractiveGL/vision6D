import pathlib
import os

from PyQt5 import QtWidgets
import numpy as np
import PIL.Image
import cv2

from ..paths_store import PathsStore
from ...widgets import VideoSampler
from ..singleton import Singleton

class VideoStore(metaclass=Singleton):

    def __init__(self):
        self.paths_store = PathsStore()

    def load_per_frame_info(self, save=False):
        if self.paths_store.video_path:
            self.video_player.cap.set(cv2.CAP_PROP_POS_FRAMES, self.paths_store.current_frame)
            ret, frame = self.video_player.cap.read()
            if ret: 
                video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.add_image(video_frame)
                if save:
                    os.makedirs(pathlib.Path(self.paths_store.video_path).parent / f"{pathlib.Path(self.paths_store.video_path).stem}_vision6D", exist_ok=True)
                    os.makedirs(pathlib.Path(self.paths_store.video_path).parent / f"{pathlib.Path(self.paths_store.video_path).stem}_vision6D" / "frames", exist_ok=True)
                    output_frame_path = pathlib.Path(self.paths_store.video_path).parent / f"{pathlib.Path(self.paths_store.video_path).stem}_vision6D" / "frames" / f"frame_{self.paths_store.current_frame}.png"
                    save_frame = PIL.Image.fromarray(video_frame)
                    
                    # save each frame
                    save_frame.save(output_frame_path)
                    self.output_text.append(f"-> Save frame {self.paths_store.current_frame}: ({self.paths_store.current_frame}/{self.video_player.frame_count})")
                    self.image_path = str(output_frame_path)

                    # save gt_pose for each frame
                    os.makedirs(pathlib.Path(self.paths_store.video_path).parent / f"{pathlib.Path(self.paths_store.video_path).stem}_vision6D" / "poses", exist_ok=True)
                    output_pose_path = pathlib.Path(self.paths_store.video_path).parent / f"{pathlib.Path(self.paths_store.video_path).stem}_vision6D" / "poses" / f"pose_{self.paths_store.current_frame}.npy"
                    self.current_pose()
                    np.save(output_pose_path, self.transformation_matrix)
                    self.output_text.append(f"-> Save frame {self.paths_store.current_frame} pose: \n{self.transformation_matrix}")
        elif self.folder_path:
            # save gt_pose for specific frame
            os.makedirs(pathlib.Path(self.folder_path) / "vision6D", exist_ok=True)
            os.makedirs(pathlib.Path(self.folder_path) / "vision6D" / "poses", exist_ok=True)
            output_pose_path = pathlib.Path(self.folder_path) / "vision6D" / "poses" / f"{pathlib.Path(self.pose_path).stem}.npy"
            self.current_pose()
            np.save(output_pose_path, self.transformation_matrix)
            self.output_text.append(f"-> Save frame {pathlib.Path(self.pose_path).stem} pose: \n{self.transformation_matrix}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
    
    def sample_video(self):
        if self.paths_store.video_path:
            self.video_sampler = VideoSampler(self.video_player, self.fps)
            res = self.video_sampler.exec_()
            if res == QtWidgets.QDialog.Accepted: self.fps = round(self.video_sampler.fps)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def play_video(self):
        if self.paths_store.video_path:
            res = self.video_player.exec_()
            if res == QtWidgets.QDialog.Accepted:
                self.paths_store.current_frame = self.video_player.current_frame
                self.output_text.append(f"-> Current frame is ({self.paths_store.current_frame}/{self.video_player.frame_count})")
                self.play_video_button.setText(f"Play ({self.paths_store.current_frame}/{self.video_player.frame_count})")
                self.load_per_frame_info()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
    
    def prev_frame(self):
        if self.paths_store.video_path:
            current_frame = self.paths_store.current_frame - self.fps
            if current_frame >= 0: 
                self.paths_store.current_frame = current_frame
                self.output_text.append(f"-> Current frame is ({self.paths_store.current_frame}/{self.video_player.frame_count})")
                pose_path = pathlib.Path(self.paths_store.video_path).parent / f"{pathlib.Path(self.paths_store.video_path).stem}_vision6D" / "poses" / f"pose_{self.paths_store.current_frame}.npy"
                if os.path.isfile(pose_path): 
                    self.transformation_matrix = np.load(pose_path)
                    self.register_pose(self.transformation_matrix)
                    self.output_text.append(f"-> Load saved frame {self.paths_store.current_frame} pose: \n{self.transformation_matrix}")
                else: self.output_text.append(f"-> No saved pose for frame {self.paths_store.current_frame}")
                self.play_video_button.setText(f"Play ({self.paths_store.current_frame}/{self.video_player.frame_count})")
                self.video_player.slider.setValue(self.paths_store.current_frame)
                self.load_per_frame_info()
        elif self.folder_path:
            if self.paths_store.current_frame > 0:
                self.paths_store.current_frame -= 1
                self.add_folder()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def next_frame(self):
        if self.paths_store.video_path:
            current_frame = self.paths_store.current_frame + self.fps
            if current_frame <= self.video_player.frame_count: 
                # save pose from the previous frame 
                self.load_per_frame_info(save=True)
                self.paths_store.current_frame = current_frame
                self.output_text.append(f"-> Current frame is ({self.paths_store.current_frame}/{self.video_player.frame_count})")
                # load pose for the current frame if the pose exist
                pose_path = pathlib.Path(self.paths_store.video_path).parent / f"{pathlib.Path(self.paths_store.video_path).stem}_vision6D" / "poses" / f"pose_{self.paths_store.current_frame}.npy"
                if os.path.isfile(pose_path): 
                    self.transformation_matrix = np.load(pose_path)
                    self.register_pose(self.transformation_matrix)
                    self.output_text.append(f"-> Load saved frame {self.paths_store.current_frame} pose: \n{self.transformation_matrix}")
                self.play_video_button.setText(f"Play ({self.paths_store.current_frame}/{self.video_player.frame_count})")
                self.video_player.slider.setValue(self.paths_store.current_frame)
                self.load_per_frame_info()
        elif self.folder_path:
            if self.paths_store.current_frame < self.total_count:
                self.paths_store.current_frame += 1
                self.add_folder()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0