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
IMAGE_PATH = DATA_DIR / "RL_20210304_0.jpg"
BACKGROUND_PATH = DATA_DIR / "black_background.jpg"

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
    return vis.App(True)
    # return vis.App(False)
    
@pytest.fixture
def configured_app(app):
    app.load_image(IMAGE_PATH, [0.01, 0.01, 1])
    app.set_reference("ossicles")
    app.load_meshes({'ossicles': OSSICLES_PATH_NO_COLOR, 'facial_nerve': FACIAL_NERVE_PATH_NO_COLOR, 'chorda': CHORDA_PATH_NO_COLOR})
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
    app.load_meshes({'sephere': pv.Sphere(radius=1)})
    app.bind_meshes("sephere", "g", ['sephere'])
    app.plot()

def test_load_mesh_from_meshfile(app):
    app.load_image(IMAGE_PATH, [0.01, 0.01, 1])
    app.set_reference("ossicles")
    app.load_meshes({'ossicles': OSSICLES_MESH_PATH, 'facial_nerve': FACIAL_NERVE_MESH_PATH, 'chorda': CHORDA_MESH_PATH})
    app.bind_meshes("ossicles", "g", ['facial_nerve', 'chorda'])
    app.bind_meshes("chorda", "h", ['ossicles', 'facial_nerve'])
    app.bind_meshes("facial_nerve", "j", ['ossicles', 'chorda'])
    app.plot()
    
def test_plot_render(configured_app):
    configured_app.plot()
    image_np = configured_app.render_scene(BACKGROUND_PATH, [0.01, 0.01, 1], False)
    image = Image.fromarray(image_np)
    image.save(DATA_DIR / "res_render_whole_plot_render_function.png")
    
def test_render_image(app):
    image_np = app.render_scene(IMAGE_PATH, [0.01, 0.01, 1], True)
    image = Image.fromarray(image_np)
    image.save(DATA_DIR / "image1.png")
    
def test_render_ossicles(app):
    app.load_meshes({'ossicles': OSSICLES_MESH_PATH, 'facial_nerve': FACIAL_NERVE_MESH_PATH, 'chorda': CHORDA_MESH_PATH})
    image_np = app.render_scene(BACKGROUND_PATH, [0.01, 0.01, 1], False, 'ossicles')
    image = Image.fromarray(image_np)
    image.save(DATA_DIR / "res_render1.png")
    # DATA_DIR / "ossicles_001_colored_not_centered.ply"
    
def test_render_whole(app):
    app.load_meshes({'ossicles': OSSICLES_MESH_PATH, 'facial_nerve': FACIAL_NERVE_MESH_PATH, 'chorda': CHORDA_MESH_PATH})
    image_np = app.render_scene(BACKGROUND_PATH, [0.01, 0.01, 1], False)
    image = Image.fromarray(image_np)
    image.save(DATA_DIR / "res_render_whole.png")

@pytest.mark.parametrize(
    "_app, expected_RT", 
    (
        (lazy_fixture("app"), np.eye(4)),
    )
)
def test_pnp_with_cube(_app, expected_RT):
    # Reference:
    # https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    
    # Create simple camera location
    _app.set_camera_extrinsics(position=(0, 0, 4),focal_point=(0, 0, 0), viewup=(0, 1, 0))
    
    # Load a cube mesh with an identify matrix pose
    cube = pv.Cube()
    
    # app.load_image(IMAGE_PATH, [0.01, 0.01, 1])
    _app.load_meshes({'cube': cube}) # Pass parameter of desired RT applied to
    _app.plot()
    
    # Create rendering
    render = _app.render_scene(BACKGROUND_PATH, (0.01, 0.01, 1), False)
    mask_render = vis.utils.color2binary_mask(vis.utils.change_mask_bg(render, [255, 255, 255], [0, 0, 0]))
    norm_mask_render = mask_render[...,0]/255
    plt.imshow(render); plt.show()  
    
    # Create 2D-3D correspondences
    pts2d, pts3d = vis.utils.create_2d_3d_pairs(norm_mask_render, render, scale=[1,1,1])
    
    logger.debug(f"The total points are {pts3d.shape[0]}")
    
    # vtkmatrix = _app.camera.GetModelViewTransformMatrix()
    # camera_matrix = pv.array_from_vtkmatrix(vtkmatrix)
    
    # homogeneous_pts3d = np.hstack((pts3d, np.ones(pts3d.shape[0]).reshape((-1,1))))
    
    # # camera_matrix = np.hstack((_app.camera_intrinsics, np.zeros(3).reshape((-1, 1))))
    
    # camera_matrix = np.array([[5., 0., 0., 0.],
    #                         [0., 5., 0., 0.],
    #                         [0., 0., 1., 0.]])
    
    # res = camera_matrix @ homogeneous_pts3d.T
       
    pts2d = pts2d.astype('float32')
    pts3d = pts3d.astype('float32')
    camera_intrinsics = _app.camera_intrinsics.astype('float32')
    
    # Solve PnP
    # cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    res = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    success, rvect, tvect, inliers = res
    
    if success:
        
        logger.debug(f"The total inliers are {len(inliers)}")
    
        # Create transformation matrix
        rmatrix = np.eye(3)
        cv2.Rodrigues(rvect, rmatrix)
        RT = np.vstack((np.hstack((rmatrix, tvect)), np.array([0,0,0,1])))
    
        logger.debug(f"\n{RT}")
        
    else:
        assert "not sucess"
        
    # Grab the corners of the Cube
    points = cube.points
    
    # Cartisian to Homogeneous
    homogeneous_points = np.hstack((points, np.ones(points.shape[0]).reshape((-1,1))))
    
    # Apply the RT
    temp = (RT @ homogeneous_points.T)
    
    # Homogeneous to Cartisian
    transformed_points = (temp[:3] / temp[3]).T
    
    # Compute the expected new location of the Cube with respect to the camera
    expected_points = points - _app.camera.position
    
    # These should be similar
