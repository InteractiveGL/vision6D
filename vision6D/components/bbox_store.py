'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: bbox_store.py
@time: 2023-07-03 20:24
@desc: create store for bbox related functions
'''

import pathlib

import cv2
import vtk
import PIL.Image
import numpy as np
import pyvista as pv

from . import Singleton
from ..tools import utils

class BboxStore(metaclass=Singleton):
    def __init__(self):
        self.reset()
        self.mirror_x = False
        self.mirror_y = False

    def reset(self):
        self.bbox_path = None
        self.bbox_actor = None
        self.bbox_opacity = 0.5

    def add_bbox(self, bbox_source, width, height):
        # default is '.npy' file
        if isinstance(bbox_source, pathlib.Path) or isinstance(bbox_source, str):
            self.bbox_path = str(bbox_source)
            bbox_source = np.load(bbox_source)

        # find the center of the image
        image_center = np.array([width // 2, height // 2, 0]) * 0.01

        if bbox_source.shape == (4, ):
            points = np.array([[bbox_source[0], bbox_source[1], 0], 
                               [bbox_source[0], bbox_source[3], 0], 
                               [bbox_source[2], bbox_source[3], 0], 
                               [bbox_source[2], bbox_source[1], 0]]) * 0.01 # x1, y1, x2, y2
        elif bbox_source.shape == (4, 3): 
            points = bbox_source
            
        if self.mirror_x: points[:, 0] = width*0.01 - points[:, 0]
        if self.mirror_y: points[:, 1] = height*0.01 - points[:, 1]
        self.bbox_bottom_point = points[np.argmax(points[:, 1])]
        self.bbox_offset = (self.bbox_bottom_point - image_center)
        cells = np.array([[2, 0, 1], [2, 1, 2], [2, 2, 3], [2, 3, 0]]).ravel()
        bbox = pv.UnstructuredGrid(cells, np.full((4,), vtk.VTK_LINE, dtype=np.uint8), points)
        bbox = bbox.translate(-self.bbox_bottom_point+self.bbox_offset, inplace=False)
        
        return bbox