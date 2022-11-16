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

OSSICLES_PATH_NO_COLOR = DATA_DIR / "ossicles_001_not_colored.ply"
FACIAL_NERVE_PATH_NO_COLOR = DATA_DIR / "facial_nerve_001_not_colored.ply"
CHORDA_PATH_NO_COLOR = DATA_DIR / "chorda_001_not_colored.ply"

OSSICLES_MESH_PATH = DATA_DIR / "5997_right_output_mesh_from_df.mesh"
FACIAL_NERVE_MESH_PATH = DATA_DIR / "5997_right_facial_nerve.mesh"
CHORDA_MESH_PATH = DATA_DIR / "5997_right_chorda.mesh"

@pytest.fixture
def app():
    return vis.App(True, IMAGE_PATH, [0.01, 0.01, 1])
    # return vis.App(False, IMAGE_PATH, OSSICLES_PATH, FACIAL_NERVE_PATH, CHORDA_PATH, [0.01, 0.01, 1])
    
@pytest.fixture
def configured_app(app):
    app.load_meshes({'ossicles': OSSICLES_PATH_NO_COLOR, 'facial_nerve': FACIAL_NERVE_PATH_NO_COLOR, 'chorda': CHORDA_PATH_NO_COLOR})
    app.set_reference("ossicles")
    app.bind_meshes("ossicles", "g", ['facial_nerve', 'chorda'])
    app.bind_meshes("chorda", "h", ['ossicles', 'facial_nerve'])
    app.bind_meshes("facial_nerve", "j", ['ossicles', 'chorda'])
    return app

def test_create_app(app):
    assert isinstance(app, vis.App)

def test_load_image(app):
    app.load_image(IMAGE_PATH, scale_factor=[0.01, 0.01, 1])
    app.plot()

def test_load_mesh_from_ply(configured_app):
    configured_app.plot()
    
def test_load_mesh_from_polydata(app):
    app.load_meshes({'sephere': pv.Sphere(radius=4)})
    app.set_reference("sephere")
    app.bind_meshes("sephere", "g", ['sephere'])
    app.plot()

def test_load_mesh_from_meshfile(app):
    app.load_meshes({'ossicles': OSSICLES_MESH_PATH, 'facial_nerve': FACIAL_NERVE_MESH_PATH, 'chorda': CHORDA_MESH_PATH})
    app.set_reference("ossicles")
    app.bind_meshes("ossicles", "g", ['facial_nerve', 'chorda'])
    app.bind_meshes("chorda", "h", ['ossicles', 'facial_nerve'])
    app.bind_meshes("facial_nerve", "j", ['ossicles', 'chorda'])
    app.plot()

def test_render_ossicles(app):
    app.render_ossicles([0.01, 0.01, 1], DATA_DIR / "ossicles_001_colored_not_centered.ply")
    
def test_plot_render(app):
    app.load_mesh(OSSICLES_PATH, FACIAL_NERVE_PATH, CHORDA_PATH)
    app.plot_render([0.01, 0.01, 1], DATA_DIR / "ossicles_001_colored_not_centered.ply")

def test_render_image(app):
    app.render_image(IMAGE_PATH, [0.01, 0.01, 1])