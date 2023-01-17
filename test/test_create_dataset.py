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
GITROOT = CWD.parent
DATA_DIR = GITROOT / 'data'

TEST_DATA_DIR = CWD / 'data'
IMAGE_JPG_PATH = TEST_DATA_DIR / "image.jpg"
IMAGE_NUMPY_PATH = TEST_DATA_DIR / "image.png"
MASK_PATH_NUMPY_FULL = TEST_DATA_DIR / "segmented_mask_numpy.npy"
MASK_PATH_NUMPY_QUARTER = TEST_DATA_DIR / "quarter_image_mask_numpy.npy"
MASK_PATH_NUMPY_SMALLEST = TEST_DATA_DIR / "smallest_image_mask_numpy.npy"
STANDARD_LENS_MASK_PATH_NUMPY = TEST_DATA_DIR / "test1.npy"

OSSICLES_PATH_NO_COLOR = TEST_DATA_DIR / "ossicles_001_not_colored.ply"
FACIAL_NERVE_PATH_NO_COLOR = TEST_DATA_DIR / "facial_nerve_001_not_colored.ply"
CHORDA_PATH_NO_COLOR = TEST_DATA_DIR / "chorda_001_not_colored.ply"

OSSICLES_MESH_PATH = TEST_DATA_DIR / "5997_right_output_mesh_from_df.mesh"
FACIAL_NERVE_MESH_PATH = TEST_DATA_DIR / "5997_right_facial_nerve.mesh"
CHORDA_MESH_PATH = TEST_DATA_DIR / "5997_right_chorda.mesh"

RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX = np.array([[ -0.14038217,  -0.3013128,   -0.94313491, -12.37503967],
                                        [  0.36747861,   0.8686707,   -0.33222081, -29.90434306],
                                        [  0.91937604,  -0.3932198,   -0.01121988, -90.78678434],
                                        [  0.,           0.,           0.,           1.        ]] ) #  GT pose

                                        
RL_20210422_0_OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.41506831,  -0.17272545,  -0.89324366, -22.39192516],
                                                        [  0.28779316,   0.95632337,  -0.05119269, -27.4517472 ],
                                                        [  0.86307206,  -0.23582096,   0.44664873, -85.09655279],
                                                        [  0.,           0.,           0.,           1.        ]]) #  GT pose

RL_20210506_0_OSSICLES_TRANSFORMATION_MATRIX = np.array([[  -0.10759918,    0.69942332,   -0.70656169,  -27.35806021],
                                                        [   0.49807515,    0.65299233,    0.5705455,   -11.57501183],
                                                        [   0.86043219,   -0.29053061,   -0.41862683, -100.00700923],
                                                        [   0.,            0.,            0.,            1.        ]] ) #  GT pose

RL_20211028_0_OSSICLES_TRANSFORMATION_MATRIX = np.array([[ -0.09361546,  -0.47387243,   0.87560326,  21.90055849],
                                                        [ -0.92587196,   0.36479044,   0.09843295,  12.69811938],
                                                        [ -0.36605634,  -0.80148166,  -0.47289524, -64.90806192],
                                                        [  0.,           0.,           0.,           1.        ]] ) #  GT pose

# full size of the (1920, 1080)
@pytest.fixture
def app_full():
    return vis.App(register=True, scale=1)
    
# 1/2 size of the (1920, 1080) -> (960, 540)
@pytest.fixture
def app_half():
    return vis.App(register=True, scale=1/2)
    
# 1/4 size of the (1920, 1080) -> (480, 270)
@pytest.fixture
def app_quarter():
    return vis.App(register=True, scale=1/4)
    
# 1/8 size of the (1920, 1080) -> (240, 135)
@pytest.fixture
def app_smallest():
    return vis.App(register=True, scale=1/8)

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
    "image_path, ossicles_path, facial_nerve_path, chorda_path, RT",
    [(DATA_DIR / "RL_20210304" / "RL_20210304_0.png", DATA_DIR / "RL_20210304" / "5997_right_output_mesh_from_df.mesh", DATA_DIR / "RL_20210304" / "5997_right_facial_nerve.mesh", DATA_DIR / "RL_20210304" / "5997_right_chorda.mesh", RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX),
     (DATA_DIR / "RL_20210422" / "RL_20210422_0.png", DATA_DIR / "RL_20210422" / "6088_right_output_mesh_from_df.mesh", DATA_DIR / "RL_20210422" / "6088_right_facial_nerve.mesh", DATA_DIR / "RL_20210422" / "6088_right_chorda.mesh", RL_20210422_0_OSSICLES_TRANSFORMATION_MATRIX),
    (DATA_DIR / "RL_20210506" / "RL_20210506_0.png", DATA_DIR / "RL_20210506" / "6108_right_output_mesh_from_df.mesh", DATA_DIR / "RL_20210506" / "6108_right_facial_nerve.mesh", DATA_DIR / "RL_20210506" / "6108_right_chorda.mesh", RL_20210506_0_OSSICLES_TRANSFORMATION_MATRIX),
    (DATA_DIR / "RL_20211028" / "RL_20211028_0.png", DATA_DIR / "RL_20211028" / "6742_left_output_mesh_from_df.mesh", DATA_DIR / "RL_20211028" / "6742_left_facial_nerve.mesh", DATA_DIR / "RL_20211028" / "6742_left_chorda.mesh", RL_20211028_0_OSSICLES_TRANSFORMATION_MATRIX),
    ]
)  
def test_load_mesh_from_dataset(image_path, ossicles_path, facial_nerve_path, chorda_path, RT):
    app_full = vis.App(register=True, scale=1)
    image_numpy = np.array(Image.open(image_path)) # (H, W, 3)
    app_full.load_image(image_numpy)
    app_full.set_transformation_matrix(RT)
    app_full.load_meshes({'ossicles': ossicles_path, 'facial_nerve': facial_nerve_path, 'chorda': chorda_path})
    app_full.bind_meshes("ossicles", "g")
    app_full.bind_meshes("chorda", "h")
    app_full.bind_meshes("facial_nerve", "j")
    app_full.set_reference("ossicles")
    app_full.plot()

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
    app.set_transformation_matrix(RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX)
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
def test_render_scene(app):
    app.register = False
    app.set_transformation_matrix(RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX)
    # app.load_meshes({'ossicles': OSSICLES_MESH_PATH, 'facial_nerve': FACIAL_NERVE_MESH_PATH, 'chorda': CHORDA_MESH_PATH})
    # image_np = app.render_scene(black_background, render_image=False, render_objects=['ossicles', 'facial_nerve', 'chorda'])
    app.load_meshes({'ossicles': OSSICLES_MESH_PATH})
    image_np = app.render_scene(render_image=False, render_objects=['ossicles'])
    plt.imshow(image_np)
    image = Image.fromarray(image_np)
    plt.imshow(image)
    plt.show()
    print("hhh")
    
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
    app.set_transformation_matrix(RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX)
    app.load_meshes({'ossicles': OSSICLES_PATH_NO_COLOR, 'facial_nerve': FACIAL_NERVE_PATH_NO_COLOR, 'chorda': CHORDA_PATH_NO_COLOR})
    app.bind_meshes("ossicles", "g")
    app.bind_meshes("chorda", "h")
    app.bind_meshes("facial_nerve", "j")
    app.set_reference("ossicles")
    image_np = app.plot()
    plt.imshow(image_np); plt.show()
    vis.utils.save_image(image_np, TEST_DATA_DIR, f"image_{name}_plot.png")
    
@pytest.mark.parametrize(
    "app, name",
    [(lazy_fixture("app_full"), "full"),
     (lazy_fixture("app_half"), "half"),
     (lazy_fixture("app_quarter"), "quarter"), 
     (lazy_fixture("app_smallest"), "smallest")]
)  
def test_generate_image(app, name):
    # image_np = app.render_scene(render_image=True, image_source=IMAGE_JPG_PATH)
    image_numpy = np.array(Image.open(IMAGE_NUMPY_PATH)) # (H, W, 3)
    image_np = app.render_scene(render_image=True, image_source=image_numpy)
    plt.imshow(image_np); plt.show()
    vis.utils.save_image(image_np, TEST_DATA_DIR, f"image_{name}.png")
     
@pytest.mark.parametrize(
    "app, name, hand_draw_mask, RT",
    [
        (lazy_fixture("app_full"), "full", MASK_PATH_NUMPY_FULL, RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX), # error: 0.28274715843164144
        (lazy_fixture("app_half"), "half", MASK_PATH_NUMPY_FULL, np.array([[ -0.14038217,  -0.3013128,   -0.94313491, -12.57631681],
                                                    [  0.36747861,   0.8686707,   -0.33222081, -29.6271648],
                                                    [  0.91937604,  -0.3932198,   -0.01121988, -121.43998767],
                                                    [  0.,           0.,           0.,           1.        ]])), # error: 0.47600086480825793
        (lazy_fixture("app_quarter"), "quarter", MASK_PATH_NUMPY_FULL, np.array([[ -0.14038217,  -0.3013128,   -0.94313491, -12.78081408],
                                                            [  0.36747861,   0.8686707,   -0.33222081, -29.76732486],
                                                            [  0.91937604,  -0.3932198,   -0.01121988, -168.66437969],
                                                            [  0.,           0.,           0.,           1.        ]])), # error: 0.06540031192830804
        (lazy_fixture("app_smallest"), "smallest", MASK_PATH_NUMPY_FULL, np.array([[ -0.14038217,  -0.3013128,   -0.94313491, -12.67572154],
                                                    [  0.36747861,   0.8686707,   -0.33222081, -29.70215918],
                                                    [  0.91937604,  -0.3932198,   -0.01121988, -355.22828667],
                                                    [  0.,           0.,           0.,           1.        ]])), # error: 2.9173443682367446
        ]
)
def test_pnp_with_masked_ossicles_surgical_microscope(app, name, RT, hand_draw_mask):
    
    mask_full = np.load(hand_draw_mask)
    
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

    # Create rendering
    render_black_bg = app.render_scene(render_image=False, render_objects=['ossicles'])
    # save the rendered whole image
    vis.utils.save_image(render_black_bg, TEST_DATA_DIR, f"rendered_mask_whole_{name}.png")

    mask_render = vis.utils.color2binary_mask(render_black_bg)
    mask_render_masked = mask_render * ossicles_mask

    render_masked_black_bg = (render_black_bg * ossicles_mask).astype(np.uint8)  # render_masked_white_bg = render_white_bg * ossicles_mask
    # save the rendered partial image
    vis.utils.save_image(render_masked_black_bg, TEST_DATA_DIR, f"rendered_mask_partial_{name}.png")
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