import ast
from functools import wraps
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

def set_pose(data, pose):
    try:
        return ast.literal_eval(data)
    except SyntaxError:
        QtWidgets.QMessageBox.warning(
                QtWidgets.QMainWindow(), 
                'vision6D', 
                "Format is not correct", 
                QtWidgets.QMessageBox.Ok, 
                QtWidgets.QMessageBox.Ok
            )
        return pose
    