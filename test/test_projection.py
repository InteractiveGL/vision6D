import logging
import pathlib
import os

import pytest
from pytest_lazyfixture import lazy_fixture
import pyvista as pv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from pytest_lazyfixture  import lazy_fixture
import skimage.transform
import torch
import trimesh
import torchvision
import vision6D as vis
from scipy.spatial import distance_matrix

import logging
import PIL

logger = logging.getLogger("vision6D")
np.set_printoptions(suppress=True)

def get_neighbors(x, y, image, pts2d):
    
    neighbors = []
    
    # Check the neighbor to the left
    if x > 0 and image[y, x-1] == 1:
        # left = (x-1, y)
        # loc = np.where((pts2d[:,0]==x-1)&(pts2d[:,1]==y))
        # neighbors.append(loc)
        neighbors.append(np.argwhere((pts2d[:,0]==x-1)&(pts2d[:,1]==y)).flatten())
    
    # Check the neighbor to the right
    if x < image.shape[1]-1 and image[y, x+1] == 1:
        # right = (x+1, y)
        # loc = np.where((pts2d[:,0]==x+1)&(pts2d[:,1]==y))
        # neighbors.append(loc)
        neighbors.append(np.argwhere((pts2d[:,0]==x+1)&(pts2d[:,1]==y)).flatten())
    
    # Check the neighbor above
    if y > 0 and image[y-1, x] == 1:
        # top = (x, y-1)
        # loc = np.where((pts2d[:,0]==x)&(pts2d[:,1]==y-1))
        # neighbors.append(loc)
        neighbors.append(np.argwhere((pts2d[:,0]==x)&(pts2d[:,1]==y-1)).flatten())
    
    # Check the neighbor below
    if y < image.shape[0]-1 and image[y+1, x] == 1:
        # buttom = (x, y+1)
        # loc = np.where((pts2d[:,0]==x)&(pts2d[:,1]==y+1))
        # neighbors.append(loc)
        neighbors.append(np.argwhere((pts2d[:,0]==x)&(pts2d[:,1]==y+1)).flatten())
    
    return neighbors

def find_closest_neighbors(pts3d):

    neighs = []
    for i in range(len(pts3d)):
        dists = np.sum((pts3d - pts3d[i])**2, axis=1)
        inds = np.argsort(dists)[:4]
        neighs.append(inds.tolist())

    neighs = np.array(neighs)

def test_draw_neibors_3d():
    pts2d = np.load(vis.config.GITROOT / "test" / "data" / "pts2d.npy")
    pts3d = np.load(vis.config.GITROOT / "test" / "data" / "pts3d_nocs.npy")

    app = vis.App(off_screen=True)

    binary_image = np.zeros((1080, 1920), dtype=np.uint8)
    color_image = np.zeros((1080, 1920, 3), dtype=np.float64)
    binary_image[pts2d[:, 1], pts2d[:, 0]] = 1
    color_image[pts2d[:, 1], pts2d[:, 0]] = pts3d

    app.set_transformation_matrix(vis.config.gt_pose_632_right)
    point_plotter = app.pv_plotter
    point_plotter.set_background('black')
    pts3d_points = point_plotter.add_points(pts3d)
    pts3d_points.user_matrix = app.transformation_matrix
    point_plotter.add_actor(pts3d_points)
    point_plotter.camera = app.camera
    point_plotter.show()
    res = point_plotter.last_image

    plt.subplot(311)
    plt.imshow(binary_image)
    plt.subplot(312)
    plt.imshow(color_image)
    plt.subplot(313)
    plt.imshow(res)
    plt.show()

    xs = pts2d[:, 0]
    ys = pts2d[:, 1]

    neighs = []
    for i in range(len(pts2d)):
        neighbors = get_neighbors(xs[i], ys[i], binary_image, pts2d)
        neighs.append(neighbors)
        
    neighs = np.array(neighs, dtype=object)

    # plot 2D vertices and lines connecting neighbors
    plt.scatter(pts3d[:, 0], pts3d[:, 1])
    for i, n in enumerate(neighs):
        for j in n:
            plt.plot([pts3d[i, 0], pts3d[j, 0]], [pts3d[i, 1], pts3d[j, 1]], 'b')

    plt.show()

    print("hhh")
    
