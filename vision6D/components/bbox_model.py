'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: bbox_model.py
@time: 2023-07-03 20:24
@desc: create store for bbox related functions
'''
from . import AbstractData
from dataclasses import dataclass


@dataclass
class BboxModel(AbstractData):
    color: str="dodgerblue"
    color_button: str=None

    def clear_attributes(self):
        super().clear_attributes()
        for field in self.__dataclass_fields__:
            if field not in AbstractData.__dataclass_fields__:
                setattr(self, field, None)