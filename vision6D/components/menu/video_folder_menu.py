import os
import pathlib

import numpy as np
import PIL.Image
from PyQt5 import QtWidgets

from ...stores import VideoStore
from ...stores import QtStore
from ...stores import MeshStore
from ...stores import ImageStore

class VideoFolderMenu():
    def __init__(self):

        # Create store
        self.qt_store = QtStore()
        self.mesh_store = MeshStore()
        self.image_store = ImageStore()
        self.video_store = VideoStore()

    def play_video(self):
        if self.video_store.video_path:
            res = self.video_store.video_player.exec_()
            if res == QtWidgets.QDialog.Accepted:
                self.video_store.current_frame = self.video_player.current_frame
                self.qt_store.output_text.append(f"-> Current frame is ({self.video_store.current_frame}/{self.video_store.total_frame})")
                video_frame = self.video_store.load_per_frame_info()
                self.image_store.add_image(video_frame)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
    
    def sample_video(self):
        if self.video_store.video_path: 
            video_sampler = self.video_store.sample_video()
            res = video_sampler.exec_()
            if res == QtWidgets.QDialog.Accepted: 
                self.video_store.fps = round(video_sampler.fps)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def delete(self):
        if self.video_store.video_path:
            self.video_store.reset()
            self.qt_store.output_text.append(f"-> Delete video {self.video_store.video_path} from vision6D")
        elif self.folder_store.folder_path:
            self.folder_store.reset()
            self.qt_store.output_text.append(f"-> Delete folder {self.folder_store.folder_path} from vision6D")

    def save_frame(self):
        if self.video_store.video_path:
            video_frame = self.video_store.load_per_frame_info()
            os.makedirs(pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D", exist_ok=True)
            os.makedirs(pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "frames", exist_ok=True)
            output_frame_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "frames" / f"frame_{self.video_store.current_frame}.png"
            save_frame = PIL.Image.fromarray(video_frame)
            
            # save each frame
            save_frame.save(output_frame_path)
            self.qt_store.output_text.append(f"-> Save frame {self.video_store.current_frame}: ({self.video_store.current_frame}/{self.video_store.total_frame})")
            self.image_store.image_path = str(output_frame_path)

            # save gt_pose for each frame
            os.makedirs(pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses", exist_ok=True)
            output_pose_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses" / f"pose_{self.video_store.current_frame}.npy"
            self.mesh_store.current_pose()
            np.save(output_pose_path, self.mesh_store.transformation_matrix)
            self.qt_store.output_text.append(f"-> Save frame {self.video_store.current_frame} pose: \n{self.mesh_store.transformation_matrix}")
        elif self.folder_store.folder_path:
            # save gt_pose for specific frame
            os.makedirs(pathlib.Path(self.folder_path) / "vision6D", exist_ok=True)
            os.makedirs(pathlib.Path(self.folder_path) / "vision6D" / "poses", exist_ok=True)
            output_pose_path = pathlib.Path(self.folder_path) / "vision6D" / "poses" / f"{pathlib.Path(self.mesh_store.pose_path).stem}.npy"
            self.mesh_store.current_pose()
            np.save(output_pose_path, self.mesh_store.transformation_matrix)
            self.qt_store.output_text.append(f"-> Save frame {pathlib.Path(self.mesh_store.pose_path).stem} pose: \n{self.mesh_store.transformation_matrix}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or a folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        return 0
        
    def prev_frame(self):
        if self.video_store.video_path:
            current_frame = self.video_store.prev_frame()
            self.qt_store.output_text.append(f"-> Current frame is ({current_frame}/{self.video_store.total_frame})")
            pose_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses" / f"pose_{current_frame}.npy"
            if os.path.isfile(pose_path): 
                self.mesh_store.transformation_matrix = np.load(pose_path)
                self.mesh_store.register_pose(self.mesh_store.transformation_matrix)
                self.qt_store.output_text.append(f"-> Load saved frame {current_frame} pose: \n{self.mesh_store.transformation_matrix}")
            else: self.qt_store.output_text.append(f"-> No saved pose for frame {current_frame}")
            self.video_store.video_player.slider.setValue(current_frame)
            video_frame = self.video_store.load_per_frame_info()
            self.image_store.add_image(video_frame)
        elif self.folder_store.folder_path:
            self.folder_store.prev_frame()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or a folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def next_frame(self):
        # save pose from the previous frame
        self.save_frame()
        if self.video_store.video_path:
            current_frame = self.video_store.next_frame()
            self.qt_store.output_text.append(f"-> Current frame is ({current_frame}/{self.video_store.total_frame})")
            # load pose for the current frame if the pose exist
            pose_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses" / f"pose_{current_frame}.npy"
            if os.path.isfile(pose_path): 
                self.mesh_store.transformation_matrix = np.load(pose_path)
                self.mesh_store.register_pose(self.mesh_store.transformation_matrix)
                self.qt_store.output_text.append(f"-> Load saved frame {current_frame} pose: \n{self.mesh_store.transformation_matrix}")
                self.video_store.video_player.slider.setValue(current_frame)
                video_frame = self.video_store.load_per_frame_info()
                self.image_store.add_image(video_frame)
        elif self.folder_store.folder_path:
            self.folder_store.next_frame()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0