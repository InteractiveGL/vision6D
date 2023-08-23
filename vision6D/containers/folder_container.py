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
import PIL.Image
from PyQt5 import QtWidgets

from ..tools import utils
from ..components import ImageStore
from ..components import MaskStore
from ..components import BboxStore
from ..components import MeshStore
from ..components import FolderStore

class FolderContainer:
    def __init__(self,
                plotter,
                play_video_button,
                toggle_register,
                add_folder,
                load_mask,
                output_text):
        
        self.plotter = plotter
        self.play_video_button = play_video_button
        self.toggle_register = toggle_register
        self.add_folder = add_folder
        self.load_mask = load_mask
        self.output_text = output_text
        
        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.bbox_store = BboxStore()
        self.mesh_store = MeshStore()
        self.folder_store = FolderStore()
  
    def save_info(self):
        if self.folder_store.folder_path:
            os.makedirs(pathlib.Path(self.folder_store.folder_path) / "vision6D", exist_ok=True)
            id = self.folder_store.current_image
            # save each image in the folder
            if self.image_store.image_actor is not None:
                os.makedirs(pathlib.Path(self.folder_store.folder_path) / "vision6D" / "images", exist_ok=True)
                output_image_path = pathlib.Path(self.folder_store.folder_path) / "vision6D" / "images" / f"{id}.png"
                image_rendered = self.image_store.render_image(camera=self.plotter.camera.copy())
                save_image = PIL.Image.fromarray(image_rendered)
                save_image.save(output_image_path)
                self.image_store.image_path = str(output_image_path)
                self.output_text.append(f"-> Save image {self.folder_store.current_image} to {str(output_image_path)}")

            if len(self.mesh_store.meshes) > 0:
                os.makedirs(pathlib.Path(self.folder_store.folder_path) / "vision6D" / "poses", exist_ok=True)
                output_pose_path = pathlib.Path(self.folder_store.folder_path) / "vision6D" / "poses" / f"{id}.npy"
                mesh_data = self.mesh_store.meshes[self.mesh_store.reference]
                self.toggle_register(mesh_data.actor.user_matrix)
                np.save(output_pose_path, mesh_data.actor.user_matrix)
                self.output_text.append(f"-> Save image {self.folder_store.current_image} pose to {str(output_pose_path)}:")
                text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                mesh_data.actor.user_matrix[0, 0], mesh_data.actor.user_matrix[0, 1], mesh_data.actor.user_matrix[0, 2], mesh_data.actor.user_matrix[0, 3], 
                mesh_data.actor.user_matrix[1, 0], mesh_data.actor.user_matrix[1, 1], mesh_data.actor.user_matrix[1, 2], mesh_data.actor.user_matrix[1, 3], 
                mesh_data.actor.user_matrix[2, 0], mesh_data.actor.user_matrix[2, 1], mesh_data.actor.user_matrix[2, 2], mesh_data.actor.user_matrix[2, 3],
                mesh_data.actor.user_matrix[3, 0], mesh_data.actor.user_matrix[3, 1], mesh_data.actor.user_matrix[3, 2], mesh_data.actor.user_matrix[3, 3])
                self.output_text.append(text)
        
            # save mask if there is a mask  
            if self.mask_store.mask_actor is not None:
                os.makedirs(pathlib.Path(self.folder_store.folder_path) / "vision6D" / "masks", exist_ok=True)
                output_mask_path = pathlib.Path(self.folder_store.folder_path) / "vision6D" / "masks" / f"{id}.png"
                mask_surface = self.mask_store.update_mask()
                self.load_mask(mask_surface)
                image = self.mask_store.render_mask(camera=self.plotter.camera.copy())
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_mask_path)
                self.mask_store.mask_path = output_mask_path
                self.output_text.append(f"-> Save image {self.folder_store.current_image} mask render to {output_mask_path}")

            # save bbox if there is a bbox  
            if self.bbox_store.bbox_actor is not None:
                os.makedirs(pathlib.Path(self.folder_store.folder_path) / "vision6D" / "bboxs", exist_ok=True)
                output_bbox_path = pathlib.Path(self.folder_store.folder_path) / "vision6D" / "bboxs" / f"{id}.npy"
                points = utils.get_bbox_actor_points(self.bbox_store.bbox_actor, self.bbox_store.image_center)
                np.save(output_bbox_path, points)
                self.bbox_store.bbox_path = output_bbox_path
                self.output_text.append(f"-> Save image {self.folder_store.current_image} bbox points to {output_bbox_path}")
        else: utils.display_warning("Need to load a folder!")
  
    def prev_info(self):
        if self.folder_store.folder_path:
            self.folder_store.prev_image()
            self.play_video_button.setText(f"Image ({self.folder_store.current_image}/{self.folder_store.total_image})")
            self.add_folder(self.folder_store.folder_path)
        else: utils.display_warning("Need to load a folder!")

    def next_info(self):
        if self.folder_store.folder_path:
            self.save_info()
            self.folder_store.next_image()
            self.play_video_button.setText(f"Image ({self.folder_store.current_image}/{self.folder_store.total_image})")
            self.add_folder(self.folder_store.folder_path)
        else: utils.display_warning("Need to load a folder!")
