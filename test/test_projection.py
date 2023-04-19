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

def test_render_point_clouds():
    # set the off_screen to True
    app = vis.App(off_screen=True, nocs_color=False)
    gt_pose = vis.config.gt_pose_5997_right
    meshpath = vis.config.OSSICLES_MESH_PATH_5997_right
    app.set_transformation_matrix(gt_pose)
    app.load_meshes({'ossicles': meshpath})
    # render the color mask since the off_screen is True
    color_mask = app.plot()

    # show the image
    plt.imshow(color_mask)
    plt.show()

    # check if the mapped color table exist in the generated color mask
    # np.any(np.all(color_mask/255 == [0, 0, 0], axis=2))

    viridis_colormap = plt.cm.get_cmap("viridis").colors

    color_mask = color_mask / 255

    binary_mask = vis.utils.color2binary_mask(color_mask)
    idx = np.where(binary_mask == 1)
    pts = np.stack((idx[1], idx[0]), axis=1)
    # Obtain the 3D verticies (normaize rgb values)
    rgb = color_mask[pts[:,1], pts[:,0]] / 255

    # get the vertices from the atlas ossicles
    # atlas_mesh = vis.utils.load_trimesh(vis.config.ATLAS_OSSICLES_MESH_PATH)
    # atlas_mesh.vertices -= np.mean(atlas_mesh.vertices, axis=0)
    mesh_5997 = vis.utils.load_trimesh(vis.config.OSSICLES_MESH_PATH_5997_right)

    # load the vertices
    # denormalize to get the rgb value for vertices respectively
    r = vis.utils.de_normalize(rgb[:, 0], mesh_5997.vertices[..., 0])
    g = vis.utils.de_normalize(rgb[:, 1], mesh_5997.vertices[..., 1])
    b = vis.utils.de_normalize(rgb[:, 2], mesh_5997.vertices[..., 2])
    vtx = np.stack([r, g, b], axis=1)

    # ~ EPNP algorithm
    pts2d = pts.astype('float32')
    pts3d = vtx.astype('float32')
    camera_intrinsics = app.camera_intrinsics.astype('float32')

    predicted_pose = np.eye(4)
    if pts2d.shape[0] > 4:
        # Use EPNP, inliers are the indices of the inliers
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, distCoeffs=np.zeros((4, 1)), confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + np.array(app.camera.position)
    
    # projected_points, _ = cv2.projectPoints(mesh_5997.vertices, rotation_vector, translation_vector, camera_matrix=app.camera_intrinsics, dist_coeffs=np.zeros((4, 1)))
    # projected_points = projected_points.reshape(-1, 2)
    # rt, _ = trimesh.registration.mesh_other(mesh_5997, atlas_mesh, samples=2454) # samples number equal to the number of vertices

    points_atlas = []
    points_vtx = []
    for i, j in zip(atlas_mesh.vertices, vtx):
        if np.isclose(i, j, rtol=0.1).all():
            points_atlas.append(i)
            points_vtx.append(j)

    print("jjj")

    # Calculate the closest points on mesh2's surface for each vertex in mesh1
    # closest_points, distance, triangle_id = atlas_mesh.nearest.on_surface(vtx)
    # # Find the index of the minimum distance
    # min_distance_index = np.argmin(distance)

    # # Get the corresponding closest points on both meshes
    # point_on_mesh2 = closest_points[min_distance_index]

    # print("jjj")

    # dist_mat = distance_matrix(atlas_mesh.vertices, vtx)
    # min_ind = dist_mat.argmin(axis=1)
    # true_vtx = vtx[min_ind, :]

def test_color_mesh_with_fast_marching():
    atlas_mesh = vis.utils.load_trimesh(vis.config.ATLAS_OSSICLES_MESH_PATH)
    mesh_5997 = vis.utils.load_trimesh(vis.config.OSSICLES_MESH_PATH_5997_right)
    distances, north_pole, south_pole = vis.utils.color_mesh_with_fast_marching(atlas_mesh)

    pl = pv.Plotter(shape=(1, 2))
    pl.subplot(0, 0)
    pl.add_mesh(atlas_mesh, scalars=distances, point_size=1, opacity=1)
    pl.add_points(atlas_mesh.vertices[north_pole], color='red', render_points_as_spheres=True, point_size=15)
    pl.add_points(atlas_mesh.vertices[south_pole], color='blue', render_points_as_spheres=True, point_size=15)
    
    pl.subplot(0, 1)
    pl.add_mesh(mesh_5997, scalars=distances, point_size=1, opacity=1)
    pl.add_points(mesh_5997.vertices[north_pole], color='red', render_points_as_spheres=True, point_size=15)
    pl.add_points(mesh_5997.vertices[south_pole], color='blue', render_points_as_spheres=True, point_size=15)
    
    pl.show()



    