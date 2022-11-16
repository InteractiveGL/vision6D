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
IMAGE_PATH = DATA_DIR / "RL_20210304_0.jpg"
OSSICLES_PATH = DATA_DIR / "ossicles_001_colored_not_centered.ply"
FACIAL_NERVE_PATH = DATA_DIR / "facial_nerve_001_colored_not_centered.ply"
CHORDA_PATH = DATA_DIR / "chorda_001_colored_not_centered.ply"


@pytest.fixture
def app():
    return vis.App(True, IMAGE_PATH, OSSICLES_PATH, FACIAL_NERVE_PATH, CHORDA_PATH, [0.01, 0.01, 1])
    # return vis.App(False, IMAGE_PATH, OSSICLES_PATH, FACIAL_NERVE_PATH, CHORDA_PATH, [0.01, 0.01, 1])

def test_create_app(app):
    assert isinstance(app, vis.App)

def test_load_image(app):
    app.load_image(IMAGE_PATH, scale_factor=[0.01, 0.01, 1])
    app.plot()

def test_load_mesh(app):
    app.load_mesh()
    app.plot()

def test_render_ossicles(app):
    app.render_ossicles([0.01, 0.01, 1], DATA_DIR / "ossicles_001_colored_not_centered.ply")
    
def test_plot_render(app):
    app.load_mesh()
    app.plot_render([0.01, 0.01, 1], DATA_DIR / "ossicles_001_colored_not_centered.ply")

def test_render_image(app):
    app.render_image(IMAGE_PATH, [0.01, 0.01, 1])