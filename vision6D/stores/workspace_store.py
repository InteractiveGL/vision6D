import json

from .singleton import Singleton
from ..stores import ImageStore
from ..stores import MaskStore
from ..stores import VideoStore
from ..stores import MeshStore

class WorkspaceStore(metaclass=Singleton):
    def __init__(self):

        self.reset()

        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.mesh_store = MeshStore()
        self.video_store = VideoStore()

    def add_workspace(self, workspace_path):
        self.workspace_path = workspace_path
        with open(str(self.workspace_path), 'r') as f: workspace = json.load(f)
        if 'image_path' in workspace: self.image_store.add_image(workspace['image_path'])
        if 'video_path' in workspace: 
            self.folder_store.folder_path = None # make sure video_path and folder_path are exclusive
            self.video_store.add_video(workspace['video_path'])
            video_frame = self.video_store.load_per_frame_info()
            self.image_store.add_image(video_frame)
        if 'mask_path' in workspace: self.mask_store.add_mask(workspace['mask_path'])
        if 'pose_path' in workspace: self.mesh_store.add_pose(workspace['pose_path'])
        if 'mesh_path' in workspace:
            mesh_path = workspace['mesh_path']
            for path in mesh_path: self.mesh_store.add_mesh(path)
            
    def reset(self):
        self.workplace_path = None