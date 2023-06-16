
from .singleton import Singleton

class PathsStore(metaclass=Singleton):

    def __init__(self):

        # Initialize file paths
        self.workspace_path = None
        self.folder_path = None
        self.video_path = None
        self.current_frame = 0
        self.image_path = None
        self.mask_path = None
        self.mesh_path = None
        self.pose_path = None

    def delete_video_folder_path(self):

        # self.video_path and self.folder_path should be exclusive
        if self.video_path: self.video_path = None
        elif self.folder_path: self.folder_path = None
        self.current_frame = 0

    def reset(self):

        self.delete_video_folder_path()

        # Initialize file paths
        self.workspace_path = None
        self.image_path = None
        self.mask_path = None
        self.mesh_path = None
        self.pose_path = None
