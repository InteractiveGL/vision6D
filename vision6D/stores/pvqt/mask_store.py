import pathlib

import pyvista as pv
import cv2
import PIL.Image
import numpy as np
from PyQt5 import QtWidgets

from ...widgets import LabelWindow
from ... import utils
from ..singleton import Singleton
from ..paths_store import PathsStore

class MaskStore(metaclass=Singleton):

    def __init__(self):

        self.mask_actor = None
        self.mask_opacity = 0.3

        self.paths_store = PathsStore()

    def draw_mask(self):
        def handle_output_path_change(output_path):
            if output_path:
                self.paths_store.mask_path = output_path
                self.add_mask(self.paths_store.mask_path)
        if self.paths_store.image_path:
            self.label_window = LabelWindow(self.paths_store.image_path)
            self.label_window.show()
            self.label_window.image_label.output_path_changed.connect(handle_output_path_change)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
    def change_mask_opacity(self, change):
        self.set_mask_opacity(self.mask_opacity + change)

    def set_mask_opacity(self, mask_opacity: float):
        assert mask_opacity>=0 and mask_opacity<=1, "image opacity should range from 0 to 1!"
        self.mask_opacity = mask_opacity
        self.mask_actor.GetProperty().opacity = mask_opacity
        self.plot_store.add_actor(self.mask_actor, pickable=True, name='mask')

    def add_mask(self, mask_source):

        if isinstance(mask_source, pathlib.WindowsPath) or isinstance(mask_source, str):
            mask_source = np.array(PIL.Image.open(mask_source), dtype='uint8')

        if mask_source.shape[-1] == 3: mask_source = cv2.cvtColor(mask_source, cv2.COLOR_RGB2GRAY)

        # Get the segmentation contour points
        contours, _ = cv2.findContours(mask_source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points2d = contours[0].squeeze()
        
        # Mirror points
        h, w = mask_source.shape[0], mask_source.shape[1]
        if self.mirror_x: points2d[:, 0] = w - points2d[:, 0]
        if self.mirror_y: points2d[:, 1] = h - points2d[:, 1]

        bottom_point = points2d[np.argmax(points2d[:, 1])]

        mask_center = (mask_source.shape[1] // 2, mask_source.shape[0] // 2)

        self.mask_offset = np.hstack(((bottom_point - mask_center)*0.01, 0))

        # Pad points a z dimension
        points = np.hstack((points2d*0.01, np.zeros(points2d.shape[0]).reshape((-1, 1))))
        # Find the bottom point on mask
        self.mask_bottom_point = points[np.argmax(points[:, 1])]

        # Create the mesh surface object
        cells = np.hstack([[points.shape[0]], np.arange(points.shape[0]), 0])
        mask_surface = pv.PolyData(points, cells).triangulate()
        mask_surface = mask_surface.translate(-self.mask_bottom_point+self.mask_offset, inplace=False)

        # Add mask surface object to the plot
        mask_mesh = self.plotter.add_mesh(mask_surface, color="white", style='surface', opacity=self.mask_opacity)
        actor, _ = self.plotter.add_actor(mask_mesh, pickable=True, name='mask')
        self.mask_actor = actor

        mask_point_data = utils.get_mask_actor_points(self.mask_actor)
        assert np.isclose(((mask_point_data+self.mask_bottom_point-self.mask_offset) - points), 0).all(), "mask_point_data and points should be equal"

    def remove_actor(self):
        mask_actor = self.mask_actor
        self.mask_actor = None
        self.mask_path = None
        return mask_actor