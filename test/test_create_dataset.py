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
IMAGE_JPG_PATH = DATA_DIR / "image.jpg"
IMAGE_NUMPY_PATH = DATA_DIR / "image.png"
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
def test_load_image(app):
    image_source = np.array(Image.open(IMAGE_NUMPY_PATH))
    # image_source = IMAGE_JPG_PATH
    app.set_image_opacity(1)
    app.load_image(image_source, scale_factor=[0.01, 0.01, 1])
    app.set_reference("image")
    app.plot()
    
@pytest.mark.parametrize(
    "app",
    [lazy_fixture("app_full"),
     lazy_fixture("app_half"),
     lazy_fixture("app_quarter"), 
     lazy_fixture("app_smallest")]
)  
def test_load_mesh(app):
    image_numpy = np.array(Image.open(IMAGE_NUMPY_PATH)) # (H, W, 3)
    app.load_image(image_numpy)
    app.set_surface_opacity(0.9)
    app.set_transformation_matrix(OSSICLES_TRANSFORMATION_MATRIX)
    app.load_meshes({'ossicles': OSSICLES_PATH_NO_COLOR, 'facial_nerve': FACIAL_NERVE_PATH_NO_COLOR, 'chorda': CHORDA_PATH_NO_COLOR})
    app.bind_meshes("ossicles", "g")
    app.bind_meshes("chorda", "h")
    app.bind_meshes("facial_nerve", "j")
    app.set_reference("ossicles")
    app.plot()
    
@pytest.mark.parametrize(
    "app, name",
    [(lazy_fixture("app_full"), "full"),
     (lazy_fixture("app_half"), "half"),
     (lazy_fixture("app_quarter"), "quarter"), 
     (lazy_fixture("app_smallest"), "smallest")]
)  
def test_save_plot(app, name):
    app.set_register(False)
    image_numpy = np.array(Image.open(IMAGE_NUMPY_PATH)) # (H, W, 3)
    app.load_image(image_numpy)
    app.set_transformation_matrix(OSSICLES_TRANSFORMATION_MATRIX)
    app.load_meshes({'ossicles': OSSICLES_PATH_NO_COLOR, 'facial_nerve': FACIAL_NERVE_PATH_NO_COLOR, 'chorda': CHORDA_PATH_NO_COLOR})
    app.bind_meshes("ossicles", "g")
    app.bind_meshes("chorda", "h")
    app.bind_meshes("facial_nerve", "j")
    app.set_reference("ossicles")
    image_np = app.plot()
    plt.imshow(image_np); plt.show()
    vis.utils.save_image(image_np, DATA_DIR, f"image_{name}_plot.png")
    
@pytest.mark.parametrize(
    "app, name",
    [(lazy_fixture("app_full"), "full"),
     (lazy_fixture("app_half"), "half"),
     (lazy_fixture("app_quarter"), "quarter"), 
     (lazy_fixture("app_smallest"), "smallest")]
)  
def test_generate_image(app, name):
    # image_np = app.render_scene(IMAGE_JPG_PATH, render_image=True)
    image_numpy = np.array(Image.open(IMAGE_NUMPY_PATH)) # (H, W, 3)
    image_np = app.render_scene(image_numpy, render_image=True)
    plt.imshow(image_np); plt.show()
    vis.utils.save_image(image_np, DATA_DIR, f"image_{name}.png")
     
@pytest.mark.parametrize(
    "app, name, RT",
    [
        (lazy_fixture("app_full"), "full", np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -11.33198707],
                                            [  0.61244989,   0.70950026,  -0.34858929, -30.4914638],
                                            [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
                                            [  0.,           0.,           0.,           1.        ]])), # error: 0.4618351490315964
        (lazy_fixture("app_half"), "half", np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -11.33198707],
                                            [  0.61244989,   0.70950026,  -0.34858929, -30.4914638],
                                            [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
                                            [  0.,           0.,           0.,           1.        ]])), # error: 0.8947398915887423
        (lazy_fixture("app_quarter"), "quarter", np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -11.33198707],
                                            [  0.61244989,   0.70950026,  -0.34858929, -30.4914638],
                                            [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
                                            [  0.,           0.,           0.,           1.        ]])), # error: 1.7811723559924346 
        (lazy_fixture("app_smallest"), "smallest", np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -11.33198707],
                                            [  0.61244989,   0.70950026,  -0.34858929, -30.4914638],
                                            [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
                                            [  0.,           0.,           0.,           1.        ]])), # error: 3.9460132718879266
        ]
)
def test_pnp_with_masked_ossicles_surgical_microscope(app, name, RT):
    
    mask_full = np.load(MASK_PATH_NUMPY_FULL)
    
    #  mask = np.load(MASK_PATH_NUMPY_FULL || MASK_PATH_NUMPY_QUARTER || MASK_PATH_NUMPY_SMALLEST) / 255
    
    if app.window_size == (1920, 1080):
        mask = mask_full / 255
        mask = np.where(mask != 0, 1, 0)
    elif app.window_size == (960, 540):
        mask = skimage.transform.rescale(mask_full, 1/2)
        mask = np.where(mask > 0.1, 1, 0) 
    elif app.window_size == (480, 270):
        mask = skimage.transform.rescale(mask_full, 1/4)
        mask = np.where(mask > 0.1, 1, 0)
    elif app.window_size == (240, 135):
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

    black_background = np.zeros((1080, 1920, 3))
    # Create rendering
    render_white_bg = app.render_scene(black_background, render_image=False, render_objects=['ossicles'])
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