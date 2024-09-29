'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: mesh_model.py
@time: 2023-07-03 20:24
@desc: create store for mesh related base functions
'''
import numpy as np
from . import AbstractData
from typing import Optional, List
from dataclasses import dataclass, field

@dataclass
class MeshModel(AbstractData):
    color: str = None
    color_button: Optional[str] = None
    texture_path: Optional[str] = None
    spacing: List[float] = field(default_factory=lambda: [1, 1, 1])
    initial_pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    undo_poses: List[np.ndarray] = field(default_factory=list)
    undo_vertices: List[np.ndarray] = field(default_factory=list)

    def clear_attributes(self):
        super().clear_attributes()
        for field in self.__dataclass_fields__:
            if field not in AbstractData.__dataclass_fields__:
                setattr(self, field, None)

