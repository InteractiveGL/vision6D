'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: image_model.py
@time: 2023-07-03 20:23
@desc: create store for image related base functions
'''
import numpy as np
from dataclasses import dataclass
from . import AbstractData
from typing import Optional

@dataclass
class ImageModel(AbstractData):
    channel: Optional[int] = None
    cx_offset: Optional[float] = 0.0
    cy_offset: Optional[float] = 0.0
    distance2camera: Optional[float] = 0.0
    center: Optional[np.ndarray] = np.array([0, 0, 0])

    def clear_attributes(self):
        super().clear_attributes()
        for field in self.__dataclass_fields__:
            if field not in AbstractData.__dataclass_fields__:
                setattr(self, field, None)

