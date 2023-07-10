'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: folder_container.py
@time: 2023-07-03 20:28
@desc: create container for folder related actions in application
'''

import os
import pathlib

import numpy as np
from PyQt5 import QtWidgets

from ..components import ImageStore
from ..components import MeshStore
from ..components import FolderStore

class FolderContainer:
    def __init__(self,
                play_video_button,
                current_pose,
                add_folder,
                output_text):
        
        self.play_video_button = play_video_button
        self.current_pose = current_pose
        self.add_folder = add_folder
        self.output_text = output_text
        
        self.image_store = ImageStore()
        self.mesh_store = MeshStore()
        self.folder_store = FolderStore()
  
    def save_info(self):
        if self.folder_store.folder_path:
            # save gt_pose for specific frame
            os.makedirs(pathlib.Path(self.folder_store.folder_path) / "vision6D", exist_ok=True)
            os.makedirs(pathlib.Path(self.folder_store.folder_path) / "vision6D" / "poses", exist_ok=True)
            output_pose_path = pathlib.Path(self.folder_store.folder_path) / "vision6D" / "poses" / f"{pathlib.Path(self.mesh_store.pose_path).stem}.npy"
            self.current_pose()
            np.save(output_pose_path, self.mesh_store.transformation_matrix)
            self.output_text.append(f"-> Save frame {pathlib.Path(self.mesh_store.pose_path).stem} pose to <span style='background-color:yellow; color:black;'>{str(output_pose_path)}</span>:")
            self.output_text.append(f"{self.mesh_store.transformation_matrix}")
            self.output_text.append(f"\n************************************************************\n")
        else: 
            QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Need to load a folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
  
    def prev_info(self):
        if self.folder_store.folder_path:
            self.folder_store.prev_frame()
            self.play_video_button.setText(f"Frame ({self.folder_store.current_frame}/{self.folder_store.total_frame})")
            self.add_folder(self.folder_store.folder_path)
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Need to load a folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def next_info(self):
        if self.folder_store.folder_path:
            self.folder_store.next_frame()
            self.play_video_button.setText(f"Frame ({self.folder_store.current_frame}/{self.folder_store.total_frame})")
            self.add_folder(self.folder_store.folder_path)
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Need to load a folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
