import ast
from functools import wraps

import numpy as np
from PyQt5 import QtWidgets

def try_except_set_spacing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                QtWidgets.QMainWindow(), 
                'vision6D', 
                "Format is not correct", 
                QtWidgets.QMessageBox.Ok, 
                QtWidgets.QMessageBox.Ok
            )
            return [1, 1, 1]
    return wrapper

@try_except_set_spacing
def set_spacing(data):
    return ast.literal_eval(data)

def set_data_format(data):
    try:
        return np.array(ast.literal_eval(data))
    except:
        QtWidgets.QMessageBox.warning(
                QtWidgets.QMainWindow(), 
                'vision6D', 
                "Format is not correct", 
                QtWidgets.QMessageBox.Ok, 
                QtWidgets.QMessageBox.Ok
            )
        return None