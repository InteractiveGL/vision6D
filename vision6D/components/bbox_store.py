'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: bbox_store.py
@time: 2023-07-03 20:24
@desc: create store for bbox related functions
'''

import vtk
import pathlib
import numpy as np
import pyvista as pv

from . import Singleton

class BboxStore(metaclass=Singleton):
    def __init__(self):
        self.reset()
        self.mirror_x = False
        self.mirror_y = False

    def reset(self):
        self.color = "dodgerblue"
        self.color_button = None
        self.bbox_path = None
        self.bbox_pv = None
        self.bbox_actor = None
        self.bbox_opacity = 0.5
        self.previous_opacity = 0.5
        self.opacity_spinbox = None

    def add_bbox(self, bbox_source, image_center, width, height):
        # default is '.npy' file
        if isinstance(bbox_source, pathlib.Path) or isinstance(bbox_source, str):
            self.bbox_path = str(bbox_source)
            bbox_source = np.load(bbox_source)

        # find the center of the image
        bbox_center = np.array([width//2, height//2, 0])

        if bbox_source.shape == (4, ): points = np.array([[bbox_source[0], bbox_source[1], 0], 
                                                        [bbox_source[0], bbox_source[3], 0], 
                                                        [bbox_source[2], bbox_source[3], 0], 
                                                        [bbox_source[2], bbox_source[1], 0]]) # x1, y1, x2, y2
        elif bbox_source.shape == (4, 3): points = bbox_source
        
        # Consider the mirror effect
        if self.mirror_x: points[:, 0] = width - points[:, 0]
        if self.mirror_y: points[:, 1] = height - points[:, 1]
        
        # Due to camera view change to right handed coordinate system
        points = points - bbox_center - image_center
        cells = np.array([[2, 0, 1], [2, 1, 2], [2, 2, 3], [2, 3, 0]]).ravel()
        self.bbox_pv = pv.UnstructuredGrid(cells, np.full((4,), vtk.VTK_LINE, dtype=np.uint8), points.astype(np.float32))
    
        return self.bbox_pv