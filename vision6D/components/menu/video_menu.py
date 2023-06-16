from functools import partial
import os
import pathlib

import numpy as np
import PIL.Image
import cv2
from PyQt5 import QtWidgets, QtGui

from ...stores import QtStore, PathsStore, VideoStore

class VideoMenu():

    def __init__(self, menu):

        # Create store
        self.paths_store = PathsStore()
        self.video_store = VideoStore()
        self.qt_store = QtStore()

        # Save parameter
        self.menu = menu
        self.menu.addAction('Play', self.video_store.play_video)
        self.menu.addAction('Sample', self.video_store.sample_video)
        self.menu.addAction('Delete', self.paths_store.delete_video_folder_path)
        self.menu.addAction('Save Frame', partial(self.video_store.load_per_frame_info, save=True))
        self.menu.addAction('Prev Frame', self.video_store.prev_frame)
        self.menu.addAction('Next Frame', self.video_store.next_frame)

        # Add shortcut to the right, left, space buttons
        QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self.qt_store.main_window).activated.connect(self.video_store.next_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self.qt_store.main_window).activated.connect(self.video_store.prev_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self.qt_store.main_window).activated.connect(self.video_store.play_video)

    def delete_video_folder(self):

        if self.paths_store.video_path:
            self.qt_store.output_text.append(f"-> Delete video {self.paths_store.video_path} from vision6D")
        elif self.folder_path:
            self.qt_store.output_text.append(f"-> Delete folder {self.paths_store.folder_path} from vision6D")

        self.paths_store.delete_video_folder_path()