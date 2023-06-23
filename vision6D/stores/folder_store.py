import os
import re
import pathlib

import numpy as np

from .singleton import Singleton
from ..stores import MeshStore
from ..stores import ImageStore
from ..stores import MaskStore

class FolderStore(metaclass=Singleton):
    def __init__(self):

        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.mesh_store = MeshStore()
        self.reset()

    def reset(self):
        self.folder_path = None
        self.current_frame = 0

    def get_files_from_folder(self, category):
        dir = pathlib.Path(self.folder_path) / category
        folders = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        if len(folders) == 1: dir = pathlib.Path(self.folder_path) / category / folders[0]
        # Retrieve files
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        self.total_count = len(files)
        # Sort files
        files.sort(key=lambda f: int(re.sub('\D', '', f)))
        return files, dir

    def add_folder(self, folder_path):
        self.folder_path = folder_path        
        folders = [d for d in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, d))]
        flag = True

        if 'images' in folders:
            flag = False
            image_files, image_dir = self.get_files_from_folder('images')
            image_path = str(image_dir / image_files[self.current_frame])
            if os.path.isfile(image_path): self.image_store.add_image(image_path)

        if 'masks' in folders:
            flag = False
            mask_files, mask_dir = self.get_files_from_folder('masks')
            mask_path = str(mask_dir / mask_files[self.current_frame])
            if os.path.isfile(mask_path): self.mask_store.add_mask(mask_path)
                
        if 'poses' in folders:
            flag = False
            pose_files, pose_dir = self.get_files_from_folder('poses')
            self.pose_path = str(pose_dir / pose_files[self.current_frame])
            if os.path.isfile(self.pose_path): self.mesh_store.add_pose_file()
                
        if self.current_frame == 0:
            if 'meshes' in folders:
                flag = False
                dir = pathlib.Path(self.folder_path) / "meshes"
                if os.path.isfile(dir / 'mesh_path.txt'):
                    with open(dir / 'mesh_path.txt', 'r') as f: mesh_path = f.read().splitlines()
                    for path in mesh_path: self.mesh_store.add_mesh(path)

        return flag

    def prev_frame(self):
        np.clip(self.current_frame, 0, self.total_count)
        self.current_frame -= 1
        self.add_folder()
        
    def next_frame(self):
        np.clip(self.current_frame, 0, self.total_count)
        self.current_frame += 1
        self.add_folder()