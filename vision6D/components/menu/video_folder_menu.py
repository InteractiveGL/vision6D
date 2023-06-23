import os
import pathlib

import numpy as np
import PIL.Image
from PyQt5 import QtWidgets

from ...stores import VideoStore
from ...stores import QtStore
from ...stores import ImageStore
from ...stores import FolderStore

class VideoFolderMenu():
    def __init__(self):

        # Create store
        self.qt_store = QtStore()
        self.image_store = ImageStore()
        self.video_store = VideoStore()
        self.folder_store = FolderStore()

    def delete(self):
        if self.video_store.video_path:
            self.video_store.reset()
            self.qt_store.output_text.append(f"-> Delete video {self.video_store.video_path} from vision6D")
        elif self.folder_store.folder_path:
            self.folder_store.reset()
            self.qt_store.output_text.append(f"-> Delete folder {self.folder_store.folder_path} from vision6D")
