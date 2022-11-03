import logging
import pathlib
import os

import pytest
import pyvista as pv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import vision6D as vis

CWD = pathlib.Path(os.path.abspath(__file__)).parent
DATA_DIR = CWD / 'data'
TEST_IMAGE_PATH = DATA_DIR / "RL_20210304_0.jpg"
TEST_MESH_PATH = DATA_DIR / "ossicles_001_colored.ply"

@pytest.fixture
def app():
    return vis.App()

def test_create_app(app):
    assert isinstance(app, vis.App)

def test_load_image(app):
    app.load_image(TEST_IMAGE_PATH)
    app.plot()

def test_load_ossicles(app):
    app.load_mesh(TEST_MESH_PATH, name="ossicles")
    app.plot()

def test_plot_image_ossicles(app):
    app.load_image(TEST_IMAGE_PATH, scaling=[0.01, 0.01, 1])
    app.load_mesh(TEST_MESH_PATH, name="ossicles")
    app.plot()
