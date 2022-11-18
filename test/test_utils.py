import logging
import pathlib
import os
import pytest
import pyvista as pv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


from pytest_lazyfixture  import lazy_fixture

import vision6D as vis

logger = logging.getLogger("vision6D")

np.set_printoptions(suppress=True)

CWD = pathlib.Path(os.path.abspath(__file__)).parent
DATA_DIR = CWD / 'data'
GREY_IMAGE = DATA_DIR / "res_render_grey.png"
RENDER_IMAGE = DATA_DIR / "res_render.png"
PLOT_IMAGE = DATA_DIR / "res_plot.png"
FRAME = DATA_DIR / "image.png"

def test_count_white_black_pixels():
    image_grey = Image.open(GREY_IMAGE)
    image_grey_np = np.array(image_grey)
    vis.utils.count_white_black_pixels(image_grey_np)
    
def test_check_pixel_in_white_image():
    image_white = Image.open(RENDER_IMAGE)
    vis.utils.check_pixel_in_image(image_white, (0, 0, 0))
    
def test_change_mask_bg():
    image_white = Image.open(RENDER_IMAGE)
    image_white_bg = np.array(image_white)
    image_black_bg = vis.utils.change_mask_bg(image_white_bg, [255, 255, 255], [0, 0, 0])
    plt.imshow(image_black_bg)
    
def test_check_pixel_in_black_image():
    image_white = Image.open(RENDER_IMAGE)
    image_white_bg = np.array(image_white)
    image_black_bg = vis.utils.change_mask_bg(image_white_bg, [255, 255, 255], [0, 0, 0])
    vis.utils.check_pixel_in_image(image_black_bg, (255, 255, 255))

def test_show_plot():
    plot = np.array(Image.open(PLOT_IMAGE))
    frame = np.array(Image.open(FRAME))
    image_white_bg = np.array(Image.open(RENDER_IMAGE))
    image_black_bg = vis.utils.change_mask_bg(image_white_bg, [255, 255, 255], [0, 0, 0])
    vis.utils.show_plot(frame, plot, image_white_bg, image_black_bg)
    
def test_color2binary_mask():
    image_white_bg = np.array(Image.open(RENDER_IMAGE))
    image_black_bg = vis.utils.change_mask_bg(image_white_bg, [255, 255, 255], [0, 0, 0])
    binary_mask = vis.utils.color2binary_mask(image_black_bg)
    logger.debug(binary_mask.shape)
    plt.imshow(binary_mask)