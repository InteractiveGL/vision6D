'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: video_container.py
@time: 2023-07-03 20:28
@desc: create container for video related actions in application
'''

import os
import pathlib

import numpy as np
import PIL.Image

from PyQt5 import QtWidgets

from ..components import ImageStore
from ..components import MaskStore
from ..components import BboxStore
from ..components import MeshStore
from ..components import VideoStore
from ..components import FolderStore

from ..tools import utils

class VideoContainer:
    def __init__(self,
                plotter,
                anchor_button,
                play_video_button,
                hintLabel, 
                toggle_register,
                add_image,
                load_mask,
                clear_plot,
                output_text):
        
        self.plotter = plotter
        self.anchor_button = anchor_button
        self.play_video_button = play_video_button
        self.hintLabel = hintLabel
        self.toggle_register = toggle_register
        self.add_image = add_image
        self.load_mask = load_mask
        self.clear_plot = clear_plot
        self.output_text = output_text
        
        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.bbox_store = BboxStore()
        self.mesh_store = MeshStore()
        self.video_store = VideoStore()
        self.folder_store = FolderStore()
  
    def add_video_file(self, video_path='', prompt=False):
        if prompt:
            video_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.avi *.mp4 *.mkv *.mov *.fly *.wmv *.mpeg *.asf *.webm)")
        if video_path:
            if self.folder_store.folder_path: self.clear_plot() # main goal is to set folder_path to None
            self.hintLabel.hide()
            self.video_store.add_video(video_path)
            self.anchor_button.setCheckable(False)
            self.anchor_button.setEnabled(False)
            self.play_video_button.setEnabled(True)
            self.play_video_button.setText(f"Play ({self.video_store.current_frame}/{self.video_store.total_frame})")
            self.output_text.append(f"-> Load video {self.video_store.video_path} into vision6D")
            self.load_per_frame_info()
            self.sample_video()

    def load_per_frame_info(self):
        video_frame = self.video_store.load_per_frame_info()
        if video_frame is not None: 
            self.add_image(video_frame)
            return video_frame
        else: 
            return None
                
    def sample_video(self):
        if self.video_store.video_path: 
            self.video_store.sample_video()
        else: utils.display_warning("Need to load a video!")

    def play_video(self):
        if self.video_store.video_path:
            self.video_store.play_video()
            self.play_video_button.setText(f"Play ({self.video_store.current_frame}/{self.video_store.total_frame})")
            self.load_per_frame_info()
        else: utils.display_warning("Need to load a video!")
    
    def save_info(self):
        if self.video_store.video_path:
            os.makedirs(pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D", exist_ok=True)
            # save each frame
            if self.image_store.image_actor is not None:
                os.makedirs(pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "frames", exist_ok=True)
                output_frame_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "frames" / f"frame_{self.video_store.current_frame}.png"
                image_rendered = self.image_store.render_image(camera=self.plotter.camera.copy())
                save_frame = PIL.Image.fromarray(image_rendered)
                save_frame.save(output_frame_path)
                self.image_store.image_path = str(output_frame_path)
                self.output_text.append(f"-> Save frame {self.video_store.current_frame} to {str(output_frame_path)}")
        
            # save gt_pose for each frame if there are any meshes
            if len(self.mesh_store.meshes) > 0:
                os.makedirs(pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses", exist_ok=True)
                output_pose_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses" / f"pose_{self.video_store.current_frame}.npy"
                mesh_data = self.mesh_store.meshes[self.mesh_store.reference]
                self.toggle_register(mesh_data.actor.user_matrix)
                np.save(output_pose_path, mesh_data.actor.user_matrix)
                self.output_text.append(f"-> Save frame {self.video_store.current_frame} pose to {str(output_pose_path)}:")
                text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                mesh_data.actor.user_matrix[0, 0], mesh_data.actor.user_matrix[0, 1], mesh_data.actor.user_matrix[0, 2], mesh_data.actor.user_matrix[0, 3], 
                mesh_data.actor.user_matrix[1, 0], mesh_data.actor.user_matrix[1, 1], mesh_data.actor.user_matrix[1, 2], mesh_data.actor.user_matrix[1, 3], 
                mesh_data.actor.user_matrix[2, 0], mesh_data.actor.user_matrix[2, 1], mesh_data.actor.user_matrix[2, 2], mesh_data.actor.user_matrix[2, 3],
                mesh_data.actor.user_matrix[3, 0], mesh_data.actor.user_matrix[3, 1], mesh_data.actor.user_matrix[3, 2], mesh_data.actor.user_matrix[3, 3])
                self.output_text.append(text)

            # save mask if there is a mask  
            if self.mask_store.mask_actor is not None:
                os.makedirs(pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "masks", exist_ok=True)
                output_mask_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "masks" / f"mask_{self.video_store.current_frame}.png"
                mask_surface = self.mask_store.update_mask()
                self.load_mask(mask_surface)
                image = self.mask_store.render_mask(camera=self.plotter.camera.copy())
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_mask_path)
                self.mask_store.mask_path = output_mask_path
                self.output_text.append(f"-> Save frame {self.video_store.current_frame} mask render to {output_mask_path}")

            # save bbox if there is a bbox  
            if self.bbox_store.bbox_actor is not None:
                os.makedirs(pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "bboxs", exist_ok=True)
                output_bbox_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "bboxs" / f"bbox_{self.video_store.current_frame}.npy"
                points = utils.get_bbox_actor_points(self.bbox_store.bbox_actor, self.bbox_store.image_center)
                np.save(output_bbox_path, points)
                self.bbox_store.bbox_path = output_bbox_path
                self.output_text.append(f"-> Save frame {self.video_store.current_frame} bbox points to {output_bbox_path}")
                
        else: utils.display_warning("Need to load a video!")

    def prev_info(self):
        if self.video_store.video_path:
            self.video_store.prev_frame()
            pose_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses" / f"pose_{self.video_store.current_frame}.npy"
            if os.path.isfile(pose_path): 
                mesh_data = self.mesh_store.meshes[self.mesh_store.reference]
                mesh_data.actor.user_matrix = np.load(pose_path)
                self.toggle_register(mesh_data.actor.user_matrix)
                self.output_text.append(f"-> Load saved frame {self.video_store.current_frame} pose:")
                text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                mesh_data.actor.user_matrix[0, 0], mesh_data.actor.user_matrix[0, 1], mesh_data.actor.user_matrix[0, 2], mesh_data.actor.user_matrix[0, 3], 
                mesh_data.actor.user_matrix[1, 0], mesh_data.actor.user_matrix[1, 1], mesh_data.actor.user_matrix[1, 2], mesh_data.actor.user_matrix[1, 3], 
                mesh_data.actor.user_matrix[2, 0], mesh_data.actor.user_matrix[2, 1], mesh_data.actor.user_matrix[2, 2], mesh_data.actor.user_matrix[2, 3],
                mesh_data.actor.user_matrix[3, 0], mesh_data.actor.user_matrix[3, 1], mesh_data.actor.user_matrix[3, 2], mesh_data.actor.user_matrix[3, 3])
                self.output_text.append(text)
            else: 
                self.output_text.append(f"-> No saved pose for frame {self.video_store.current_frame}")
            self.play_video_button.setText(f"Play ({self.video_store.current_frame}/{self.video_store.total_frame})")
            self.load_per_frame_info()
        else: utils.display_warning("Need to load a video!")

    def next_info(self):
        if self.video_store.video_path:
            self.save_info()
            self.video_store.next_frame()
            # load pose for the current frame if the pose exist
            pose_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses" / f"pose_{self.video_store.current_frame}.npy"
            if os.path.isfile(pose_path):
                self.toggle_register(np.load(pose_path))
                self.output_text.append(f"-> Load saved frame {self.video_store.current_frame} pose:")
                mesh_data = self.mesh_store.meshes[self.mesh_store.reference]
                text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                mesh_data.actor.user_matrix[0, 0], mesh_data.actor.user_matrix[0, 1], mesh_data.actor.user_matrix[0, 2], mesh_data.actor.user_matrix[0, 3], 
                mesh_data.actor.user_matrix[1, 0], mesh_data.actor.user_matrix[1, 1], mesh_data.actor.user_matrix[1, 2], mesh_data.actor.user_matrix[1, 3], 
                mesh_data.actor.user_matrix[2, 0], mesh_data.actor.user_matrix[2, 1], mesh_data.actor.user_matrix[2, 2], mesh_data.actor.user_matrix[2, 3],
                mesh_data.actor.user_matrix[3, 0], mesh_data.actor.user_matrix[3, 1], mesh_data.actor.user_matrix[3, 2], mesh_data.actor.user_matrix[3, 3])
                self.output_text.append(text)
            self.play_video_button.setText(f"Play ({self.video_store.current_frame}/{self.video_store.total_frame})")
            self.load_per_frame_info()
        else: utils.display_warning("Need to load a video!")
