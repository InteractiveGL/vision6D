'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: folder_store.py
@time: 2023-07-03 20:23
@desc: create store for folder related base functions
'''

import os
import re
import pathlib

import numpy as np

from . import Singleton

class FolderStore(metaclass=Singleton):
    def __init__(self):
        self.reset()

    def reset(self):
        self.folder_path = None
        self.current_image = 0

    def get_files_from_folder(self, category):
        dir = pathlib.Path(self.folder_path) / category
        folders = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        if len(folders) == 1: dir = pathlib.Path(self.folder_path) / category / folders[0]
        # Retrieve files
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        self.total_image = len(files)
        # Sort files
        files.sort(key=lambda f: int(re.sub('\D', '', f)))
        return files, dir

    def add_folder(self, folder_path, meshes):
        self.folder_path = folder_path
        folders = [d for d in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, d))] 
        image_path = ''
        mask_path = ''
        pose_path = ''
        mesh_path = ''
        if 'images' in folders:
            image_files, image_dir = self.get_files_from_folder('images')
            path = str(image_dir / image_files[self.current_image])
            if os.path.isfile(path): image_path = path
        if 'masks' in folders:
            mask_files, mask_dir = self.get_files_from_folder('masks')
            path = str(mask_dir / mask_files[self.current_image])
            if os.path.isfile(path): mask_path = path
        if 'poses' in folders:
            pose_files, pose_dir = self.get_files_from_folder('poses')
            path = str(pose_dir / pose_files[self.current_image])
            if os.path.isfile(path): pose_path = path
        if self.current_image == 0 or len(meshes) == 0:
            if 'meshes' in folders:
                dir = pathlib.Path(self.folder_path) / "meshes"
                path = dir / 'mesh_path.txt'
                if os.path.isfile(path): mesh_path = path

        return image_path, mask_path, pose_path, mesh_path
    
    def prev_image(self):
        self.current_image -= 1
        self.current_image = np.clip(self.current_image, 0, self.total_image)
        
    def next_image(self):
        self.current_image += 1
        self.current_image = np.clip(self.current_image, 0, self.total_image)