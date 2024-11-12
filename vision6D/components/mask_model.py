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

@dataclass
class MaskModel(AbstractData):
    color: str="white"
    color_button: str=None

    def clear_attributes(self):
        super().clear_attributes()
        for field in self.__dataclass_fields__:
            if field not in AbstractData.__dataclass_fields__:
                setattr(self, field, None)