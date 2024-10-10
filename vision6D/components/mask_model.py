'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: mask_model.py
@time: 2023-07-03 20:24
@desc: create store for mask related functions
'''

from dataclasses import dataclass
from . import AbstractData
import numpy as np
from typing import Optional

@dataclass
class MaskModel(AbstractData):
    color: str="white"
    mask_center: Optional[np.ndarray] = np.array([0, 0, 0])
    image_center: Optional[np.ndarray] = np.array([0, 0, 0])
    color_button: str=None

    def clear_attributes(self):
        super().clear_attributes()
        for field in self.__dataclass_fields__:
            if field not in AbstractData.__dataclass_fields__:
                setattr(self, field, None)