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

@pytest.fixture
def app():
    # return vis.App(True, TEST_IMAGE_PATH, [0.01, 0.01, 1])
    return vis.App(False, TEST_IMAGE_PATH, [0.01, 0.01, 1])

def test_create_app(app):
    assert isinstance(app, vis.App)

def test_load_image(app):
    app.load_image(TEST_IMAGE_PATH, scale_factor=[0.01, 0.01, 1])
    app.plot()

def test_load_mesh(app):
    app.load_mesh(DATA_DIR / "ossicles_001_colored_not_centered.ply", name="ossicles", rgb=True)
    app.load_mesh(DATA_DIR / "facial_nerve_001_colored_not_centered.ply", name="facial_nerve", rgb=True)
    app.load_mesh(DATA_DIR / "chorda_001_colored_not_centered.ply", name="chorda", rgb=True)
    app.plot()

def test_render_ossicles(app):
    app.render_ossicles([0.01, 0.01, 1], DATA_DIR / "ossicles_001_colored_not_centered.ply", rgb=True)
    
def test_plot_render(app):
    app.load_mesh(DATA_DIR / "ossicles_001_colored_not_centered.ply", name="ossicles", rgb = True)
    app.load_mesh(DATA_DIR / "facial_nerve_001_colored_not_centered.ply", name="facial_nerve", rgb = True)
    app.load_mesh(DATA_DIR / "chorda_001_colored_not_centered.ply", name="chorda", rgb = True)
    app.plot_render([0.01, 0.01, 1], DATA_DIR / "ossicles_001_colored_not_centered.ply")

def test_render_image(app):
    app.render_image(TEST_IMAGE_PATH, [0.01, 0.01, 1])