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
from dataclasses import dataclass
from typing import Optional, Dict

from . import Singleton

@dataclass
class BboxData:
    name: str=None
    bbox_path: str=None
    bbox_source: str=None
    bbox_pv: pv.UnstructuredGrid=None
    actor: Optional[pv.Actor] = None
    opacity: float=0.5
    previous_opacity: float=0.5
    opacity_spinbox: Optional[str]=None
    mirror_x: bool=False
    mirror_y: bool=False
    color: str="dodgerblue"

class BboxStore(metaclass=Singleton):
    def __init__(self):
        self.reference: Optional[str] = None
        self.bboxes: Dict[str, BboxData] = {}
        self.bbox_data = BboxData()

    def reset(self):
        pass

    def add_bbox(self, bbox_source, image_center, width, height):
        # default is '.npy' file
        if isinstance(bbox_source, pathlib.Path) or isinstance(bbox_source, str):
            bbox_path = str(bbox_source)
            name = pathlib.Path(bbox_path).stem
            bbox_source = np.load(bbox_source)
        
        # find the center of the image
        bbox_center = np.array([width//2, height//2, 0])

        if bbox_source.shape == (4, ): points = np.array([[bbox_source[0], bbox_source[1], 0], [bbox_source[0], bbox_source[3], 0], [bbox_source[2], bbox_source[3], 0], [bbox_source[2], bbox_source[1], 0]]) # x1, y1, x2, y2
        elif bbox_source.shape == (4, 3): points = bbox_source
        
        # Consider the mirror effect
        if self.bbox_data.mirror_x: points[:, 0] = width - points[:, 0]
        if self.bbox_data.mirror_y: points[:, 1] = height - points[:, 1]
        
        # Due to camera view change to right handed coordinate system
        points = points - bbox_center - image_center
        cells = np.array([[2, 0, 1], [2, 1, 2], [2, 2, 3], [2, 3, 0]]).ravel()
        bbox_pv = pv.UnstructuredGrid(cells, np.full((4,), vtk.VTK_LINE, dtype=np.uint8), points.astype(np.float32))
        bbox_data = BboxData(name=name,
                            bbox_path=bbox_path,
                            bbox_source=bbox_source,
                            bbox_pv=bbox_pv, 
                            actor=None, 
                            opacity=0.5, 
                            previous_opacity=0.5, 
                            opacity_spinbox=None, 
                            mirror_x=False, 
                            mirror_y=False, 
                            color="dodgerblue")
        self.bboxes[name] = bbox_data
        return bbox_data