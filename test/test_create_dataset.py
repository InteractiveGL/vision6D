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

import vision6D as vis

logger = logging.getLogger("vision6D")

np.set_printoptions(suppress=True)

CWD = pathlib.Path(os.path.abspath(__file__)).parent
DATA_DIR = CWD / 'data'
IMAGE_PATH = DATA_DIR / "image.jpg"
BACKGROUND_PATH = DATA_DIR / "black_background.jpg"
MASK_PATH_NUMPY_FULL = DATA_DIR / "segmented_mask_numpy.npy"
MASK_PATH_NUMPY_QUARTER = DATA_DIR / "quarter_image_mask_numpy.npy"
MASK_PATH_NUMPY_SMALLEST = DATA_DIR / "smallest_image_mask_numpy.npy"
STANDARD_LENS_MASK_PATH_NUMPY = DATA_DIR / "test1.npy"

OSSICLES_PATH = DATA_DIR / "ossicles_001_colored_not_centered.ply"
FACIAL_NERVE_PATH = DATA_DIR / "facial_nerve_001_colored_not_centered.ply"
CHORDA_PATH = DATA_DIR / "chorda_001_colored_not_centered.ply"

OSSICLES_PATH_NO_COLOR = DATA_DIR / "ossicles_001_not_colored.ply"
FACIAL_NERVE_PATH_NO_COLOR = DATA_DIR / "facial_nerve_001_not_colored.ply"
CHORDA_PATH_NO_COLOR = DATA_DIR / "chorda_001_not_colored.ply"

OSSICLES_MESH_PATH = DATA_DIR / "5997_right_output_mesh_from_df.mesh"
FACIAL_NERVE_MESH_PATH = DATA_DIR / "5997_right_facial_nerve.mesh"
CHORDA_MESH_PATH = DATA_DIR / "5997_right_chorda.mesh"

OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -11.33198707],
                                            [  0.61244989,   0.70950026,  -0.34858929, -30.4914638],
                                            [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
                                            [  0.,           0.,           0.,           1.        ]]) #  GT pose

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.0955,  -0.5021,  -0.8595, -11.2299],
#                                         [  0.6166,   0.7077,  -0.3449, -30.2564],
#                                         [  0.7815,  -0.4970,   0.3772, -15.5285],
#                                         [  0.0000,   0.0000,   0.0000,   1.0000]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[     0.0483,     -0.4848,     -0.8733,     -9.4425],
#          [     0.5884,      0.7203,     -0.3673,    -29.5993],
#          [     0.8071,     -0.4961,      0.3201,   2674.1137],
#          [     0.0000,      0.0000,      0.0000,      1.0000]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  1,  0,  0,  0],
#                                             [  0,  1,  0, 0],
#                                             [  0,  0,  1,  500],
#                                             [  0., 0.,  0., 1. ]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.08788493,  -0.49934587,  -0.86193385, -11.49899185],
#                                         [  0.61244989,   0.70950026,  -0.34858929, -30.56813355],
#                                         [  0.78560891,  -0.49725556,   0.3681787,  500.        ],
#                                         [  0.,           0.,           0.,           1.        ]])

# OSSICLES_TRANSFORMATION_MATRIX =np.array([[     0.1799,     -0.5369,     -0.8242,     -8.8171],
#          [     0.7707,      0.5977,     -0.2211,    -26.8200],
#          [     0.6113,     -0.5954,      0.5213,   2415.0381],
#          [     0.0000,      0.0000,      0.0000,      1.0000]])

# OSSICLES_TRANSFORMATION_MATRIX =np.array([[     0.1079,     -0.5122,     -0.8520,    -11.1576],
#          [     0.6046,      0.7141,     -0.3528,    -30.5566],
#          [     0.7892,     -0.4771,      0.3867,    486.7139],
#          [     0.0000,      0.0000,      0.0000,      1.0000]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  -0.01697852,   -0.38156652,   -0.92418544,  -13.29829586],
#                                         [   0.58539971,    0.74553668,   -0.31856276,  -30.33314405],
#                                         [   0.81056703,   -0.54642662,    0.21071082, 4000.71245848],
#                                         [   0.,            0.,            0.,            1.        ]])


# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  -0.03696289,   -0.45066371,   -0.89192823,  -10.77123474],
#                                         [   0.61518516,    0.69310679,   -0.37569962,  -30.43366921],
#                                         [   0.78751569,   -0.56258795,    0.25162239, 1985.80919978],
#                                         [   0.,            0.,            0.,            1.        ]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[     0.7565,     -0.5291,      0.3845,     16.1959],
#          [     0.5796,      0.8147,     -0.0193,    -18.4053],
#          [    -0.3031,      0.2375,      0.9229,   -438.4466],
#          [     0.0000,      0.0000,      0.0000,      1.0000]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[     0.0929,     -0.5055,     -0.8578,     -0.9356],
#                                     [     0.6765,      0.6642,     -0.3181,    -24.1057],
#                                     [     0.7306,     -0.5507,      0.4037,   -457.9619],
#                                     [     0.0000,      0.0000,      0.0000,      1.0000]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.0900,  -0.4999,  -0.8614,  -1.6455],
#                                     [  0.6185,   0.7060,  -0.3451, -24.9873],
#                                     [  0.7806,  -0.5017,   0.3728,  -7.1757],
#                                     [  0.0000,   0.0000,   0.0000,   1.0000]]) #  predicted from pytorch3d


# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.37177922,   0.8300953,   -0.41559835, -24.30279375],
#                                             [  0.51533184,  -0.5569185,   -0.65136388,  -4.5669351],
#                                             [ -0.7721485,    0.0279925,   -0.63482527,  -3.57181275],
#                                             [  0.,           0.,           0.,           1.,        ]]) #  GT pose

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[     0.3468,      0.8284,     -0.4399,    -24.6587],
#                                         [     0.4898,     -0.5599,     -0.6682,     -4.6086],
#                                         [    -0.7998,      0.0163,     -0.6000,     26.7000],
#                                         [     0.0000,      0.0000,      0.0000,      1.0000]]) #  predicted from pytorch3d

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -2.28768117],
#                                         [  0.61244989,   0.70950026,  -0.34858929, -25.39078897],
#                                         [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
#                                         [  0.,           0.,           0.,           1.        ]]) #  GT pose

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.0869,  -0.5010,  -0.8611,  -2.0267],
#                                         [  0.6084,   0.7111,  -0.3524, -25.3319],
#                                         [  0.7888,  -0.4933,   0.3667, -16.5832],
#                                         [  0.0000,   0.0000,   0.0000,   1.0000]]) # predicted from pytorch3d

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.08725841,  -0.49920268,  -0.86208042,  -1.773618  ],
#                                 [  0.61232186,   0.7094788 ,  -0.34885781, -25.13447245],
#                                 [  0.78577854,  -0.49742991,   0.3675807 ,   2.70771307],
#                                 [  0.        ,   0.        ,   0.        ,   1.        ]]) # GT pose

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.0926,  -0.5013,  -0.8603,  -1.6665],
#                             [  0.6176,   0.7067,  -0.3453, -25.0299],
#                             [  0.7811,  -0.4993,   0.3750,  -5.6169],
#                             [  0.0000,   0.0000,   0.0000,   1.0000]]) # predicted pose

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.08771557,  -0.49943043,  -0.8619021 ,  -1.75110001],
#                                 [  0.61228039,   0.70952996,  -0.34882654, -25.11469416],
#                                 [  0.78575995,  -0.49712824,   0.36802828,   1.49357594],
#                                 [  0.        ,   0.        ,   0.        ,   1.        ]]) # GT pose

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.0905,  -0.5002,  -0.8611,  -1.6447],
#                                     [  0.6181,   0.7062,  -0.3453, -25.0012],
#                                     [  0.7809,  -0.5010,   0.3731,  -7.7220],
#                                     [  0.0000,   0.0000,   0.0000,   1.0000]]) # predicted pose

# full size of the (1920, 1080)
@pytest.fixture
def app_full():
    return vis.App(register=True,
                   scale=1)
    
# 1/2 size of the (1920, 1080) -> (960, 540)
@pytest.fixture
def app_half():
    return vis.App(register=True,
                   scale=1/2)
    
# 1/4 size of the (1920, 1080) -> (480, 270)
@pytest.fixture
def app_quarter():
    return vis.App(register=True,
                   scale=1/4)
    
# 1/8 size of the (1920, 1080) -> (240, 135)
@pytest.fixture
def app_smallest():
    return vis.App(register=True,
                   scale=1/8)
    
@pytest.mark.parametrize(
    "app",
    [lazy_fixture("app_full"),
     lazy_fixture("app_half"),
     lazy_fixture("app_quarter"), 
     lazy_fixture("app_smallest")]
)  
def test_load_mesh(app):
    app.load_image(IMAGE_PATH, [0.01, 0.01, 1])
    app.set_transformation_matrix(OSSICLES_TRANSFORMATION_MATRIX)
    app.load_meshes({'ossicles': OSSICLES_PATH_NO_COLOR, 'facial_nerve': FACIAL_NERVE_PATH_NO_COLOR, 'chorda': CHORDA_PATH_NO_COLOR})
    app.bind_meshes("ossicles", "g")
    app.bind_meshes("chorda", "h")
    app.bind_meshes("facial_nerve", "j")
    app.set_reference("ossicles")
    app.plot()
    
@pytest.mark.parametrize(
    "app",
    [lazy_fixture("app_full"), 
     lazy_fixture("app_half"),
     lazy_fixture("app_quarter"), 
     lazy_fixture("app_smallest")]
)  
def test_generate_image(app):
    if app.window_size == (1920, 1080):
        name = "full"
    elif app.window_size == (960, 540):
        name = "half"
    elif app.window_size == (480, 270):
        name = "quarter"
    elif app.window_size == (240, 135):
        name = "smallest"
    image_np = app.render_scene(IMAGE_PATH, [0.01, 0.01, 1], True)
    vis.utils.save_image(image_np, DATA_DIR, f"image_{name}.png")
    plt.imshow(image_np); plt.show()
     
@pytest.mark.parametrize(
    "app, RT",
    [
        (lazy_fixture("app_full"), np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -11.33198707],
                                            [  0.61244989,   0.70950026,  -0.34858929, -30.4914638],
                                            [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
                                            [  0.,           0.,           0.,           1.        ]])), # error: 0.4618351490315964
        (lazy_fixture("app_half"), np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -11.33198707],
                                            [  0.61244989,   0.70950026,  -0.34858929, -30.4914638],
                                            [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
                                            [  0.,           0.,           0.,           1.        ]])), # error: 0.8947398915887423
        (lazy_fixture("app_quarter"), np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -11.33198707],
                                            [  0.61244989,   0.70950026,  -0.34858929, -30.4914638],
                                            [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
                                            [  0.,           0.,           0.,           1.        ]])), # error: 1.7811723559924346 
        (lazy_fixture("app_smallest"), np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -11.33198707],
                                            [  0.61244989,   0.70950026,  -0.34858929, -30.4914638],
                                            [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
                                            [  0.,           0.,           0.,           1.        ]])), # error: 3.9460132718879266
        ]
)
def test_pnp_with_masked_ossicles_surgical_microscope(app, RT):
    
    mask_full = np.load(MASK_PATH_NUMPY_FULL)
    
    #  mask = np.load(MASK_PATH_NUMPY_FULL || MASK_PATH_NUMPY_QUARTER || MASK_PATH_NUMPY_SMALLEST) / 255
    
    if app.window_size == (1920, 1080):
        name = "full"
        mask = mask_full / 255
        mask = np.where(mask != 0, 1, 0)
    elif app.window_size == (960, 540):
        name = "half"
        mask = skimage.transform.rescale(mask_full, 1/2)
        mask = np.where(mask > 0.1, 1, 0) 
    elif app.window_size == (480, 270):
        name = "quarter"
        mask = skimage.transform.rescale(mask_full, 1/4)
        mask = np.where(mask > 0.1, 1, 0)
    elif app.window_size == (240, 135):
        name = "smallest"
        mask = skimage.transform.rescale(mask_full, 1/8)
        mask = np.where(mask > 0.1, 1, 0)
        
    plt.subplot(211)
    plt.imshow(mask_full)
    plt.subplot(212)
    plt.imshow(mask)
    plt.show()
    
    # Use whole mask with epnp
    # mask = np.ones((1080, 1920))
   
    # expand the dimension
    ossicles_mask = np.expand_dims(mask, axis=-1)
   
    # Dilate mask
    # mask = cv2.dilate(mask, np.ones((5,5),np.uint8), iterations = 100)
        
    app.set_transformation_matrix(RT)
    app.load_meshes({'ossicles': OSSICLES_PATH_NO_COLOR})
    app.plot()
    
    # Create rendering
    render_white_bg = app.render_scene(BACKGROUND_PATH, (0.01, 0.01, 1), render_image=False, render_objects=['ossicles'])
    render_black_bg = vis.utils.change_mask_bg(render_white_bg, [255, 255, 255], [0, 0, 0])
    # save the rendered whole image
    vis.utils.save_image(render_black_bg, DATA_DIR, f"rendered_mask_whole_{name}.png")
    mask_render = vis.utils.color2binary_mask(render_black_bg)
    mask_render_masked = mask_render * ossicles_mask
    render_masked_black_bg = (render_black_bg * ossicles_mask).astype(np.uint8)  # render_masked_white_bg = render_white_bg * ossicles_mask
    # save the rendered partial image
    vis.utils.save_image(render_masked_black_bg, DATA_DIR, f"rendered_mask_partial_{name}.png")
    assert (mask_render_masked == vis.utils.color2binary_mask(render_masked_black_bg)).all()
    
    plt.subplot(221)
    plt.imshow(render_black_bg)
    plt.subplot(222)
    plt.imshow(ossicles_mask)
    plt.subplot(223)
    plt.imshow(mask_render_masked)
    plt.subplot(224)
    plt.imshow(render_masked_black_bg)
    plt.show()
    
    # Create 2D-3D correspondences
    pts2d, pts3d = vis.utils.create_2d_3d_pairs(mask_render_masked, render_masked_black_bg, app, 'ossicles')
    
    logger.debug(f"The total points are {pts3d.shape[0]}")

    pts2d = pts2d.astype('float32')
    pts3d = pts3d.astype('float32')
    camera_intrinsics = app.camera_intrinsics.astype('float32')
    
    if pts2d.shape[0] < 4:
        predicted_pose = np.eye(4)
        inliers = []
    else:
        dist_coeffs = np.zeros((4, 1))
        
        # Get a rotation matrix
        predicted_pose = np.eye(4)
        
        # Use EPNP
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
            
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + np.array(app.camera.position)
            logger.debug(len(inliers)) # 50703
            
    logger.debug(f"\ndifference from predicted pose and RT pose: {np.sum(np.abs(predicted_pose - RT))}")
            
    assert np.isclose(predicted_pose, RT, atol=4).all()