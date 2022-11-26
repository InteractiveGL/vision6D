import logging
import pathlib
import os

import pytest
import pyvista as pv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation as R
from pytest_lazyfixture  import lazy_fixture

import vision6D as vis

logger = logging.getLogger("vision6D")

np.set_printoptions(suppress=True)

def test_read_label():
    with open('project-1-at-2022-11-26-09-48-2d5d27ff.json') as json_file:
        labels = json.load(json_file)
        
    points = np.array(labels[0]["annotations"][0]["result"][0]["value"]["points"])
    
    image = np.zeros((1080, 1920)).astype('float32')
    
    res = np.ma.masked_values(image, points)