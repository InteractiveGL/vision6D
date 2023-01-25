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

OSSICLES_MESH_PATH = DATA_DIR / "5997_right_ossicles.mesh"
FACIAL_NERVE_MESH_PATH = DATA_DIR / "5997_right_facial_nerve.mesh"
CHORDA_MESH_PATH = DATA_DIR / "5997_right_chorda.mesh"

# OSSICLES_MESH_PATH_PLY = TEST_DATA_DIR / "ossicles_001_not_colored.ply"
# OLD_OSSICLES_MESH_PATH = DATA_DIR / "5997_right_output_mesh_from_df.mesh"
# OSSICLES_MESH_PATH = TEST_DATA_DIR / "ossicles_001_not_colored.ply"
# FACIAL_NERVE_MESH_PATH = TEST_DATA_DIR / "facial_nerve_001_not_colored.ply"
# CHORDA_MESH_PATH = TEST_DATA_DIR / "chorda_001_not_colored.ply"

OSSICLES_MESH_PATH_5997 = DATA_DIR / "5997_right_ossicles.mesh"
OSSICLES_MESH_PATH_6088 = DATA_DIR / "6088_right_ossicles.mesh"
OSSICLES_MESH_PATH_6108 = DATA_DIR / "6108_right_ossicles.mesh"
OSSICLES_MESH_PATH_6742 = DATA_DIR / "6742_left_ossicles.mesh"

mask_5997_hand_draw_numpy = DATA_DIR / "mask_5997_hand_draw_numpy.npy"
mask_6088_hand_draw_numpy = DATA_DIR / "mask_6088_hand_draw_numpy.npy"
mask_6108_hand_draw_numpy = DATA_DIR / "mask_6108_hand_draw_numpy.npy"
mask_6742_hand_draw_numpy = DATA_DIR / "mask_6742_hand_draw_numpy.npy"

gt_pose_5997 = np.array([[  -0.14498174,   -0.34676691,   -0.92667849,  -10.60345244],
                    [   0.33413722,    0.86439266,   -0.3757361,   -29.9112112 ],
                    [   0.93130693,   -0.36411267,   -0.00945343, -119.95253896],
                    [   0.,            0.,            0.,            1.        ]] ) #  GT pose
                                        
gt_pose_60881 = np.array([[  0.2720979,   -0.10571448,  -0.96757074, -23.77536492],
                                                        [  0.33708389,   0.95272971,  -0.00929905, -27.7404459 ],
                                                        [  0.91309533,  -0.32021318,   0.29176419, -20.44275252],
                                                        [  0.,           0.,           0.,           1.        ]]) #  GT pose


gt_pose_6088 = np.array([[  0.36049218,  -0.12347807,  -0.93605796, -24.3777202 ],
                                                        [  0.31229879,   0.96116227,  -0.00651795, -27.43646144],
                                                        [  0.89102231,  -0.28692541,   0.38099733, -19.1631882 ],
                                                        [  0.,           0.,           0.,           1.        ]])


gt_pose_6108 = np.array([[  0.18750646,   0.35092506,  -0.91743823, -29.81739935],
                                                    [  0.55230585,   0.73470634,   0.39390969, -19.10118172],
                                                    [  0.81228048,  -0.58056712,  -0.0560558,  227.40282413],
                                                    [  0.,           0.,           0.,           1.        ]]) #  GT pose

gt_pose_6742 = np.array([[ -0.00205008,  -0.27174699,   0.96236655,  16.14180134],
                                                        [ -0.4431008,    0.86298269,   0.24273971,  -4.42885807],
                                                        [ -0.89646944,  -0.42592774,  -0.1221805,  458.83536963],
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
    [(DATA_DIR / "5997_0.png", DATA_DIR / "5997_right_ossicles.mesh", DATA_DIR / "5997_right_facial_nerve.mesh", DATA_DIR / "5997_right_chorda.mesh", gt_pose_5997),
    (DATA_DIR / "6088_0.png", DATA_DIR / "6088_right_ossicles.mesh", DATA_DIR / "6088_right_facial_nerve.mesh", DATA_DIR / "6088_right_chorda.mesh", gt_pose_6088),
    (DATA_DIR / "6108_0.png", DATA_DIR / "6108_right_ossicles.mesh", DATA_DIR / "6108_right_facial_nerve.mesh", DATA_DIR / "6108_right_chorda.mesh", gt_pose_6108),
    (DATA_DIR / "6742_0.png", DATA_DIR / "6742_left_ossicles.mesh", DATA_DIR / "6742_left_facial_nerve.mesh", DATA_DIR / "6742_left_chorda.mesh", gt_pose_6742),
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
    app.set_transformation_matrix(gt_pose_5997)
    app.load_meshes({'ossicles': OSSICLES_MESH_PATH, 'facial_nerve': FACIAL_NERVE_MESH_PATH, 'chorda': CHORDA_MESH_PATH})
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
    app.set_transformation_matrix(gt_pose_5997)
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
    app.set_transformation_matrix(gt_pose_5997)
    app.load_meshes({'ossicles': OSSICLES_MESH_PATH, 'facial_nerve': FACIAL_NERVE_MESH_PATH, 'chorda': CHORDA_MESH_PATH})
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
    "app, name, hand_draw_mask, ossicles_path, RT",
    [
        (lazy_fixture("app_full"), "5997",  mask_5997_hand_draw_numpy, OSSICLES_MESH_PATH_5997, gt_pose_5997), # error: 0.8573262508172124
        (lazy_fixture("app_full"), "6088", mask_6088_hand_draw_numpy, OSSICLES_MESH_PATH_6088, gt_pose_6088), # error: 5.253714309928259
        (lazy_fixture("app_full"), "6108", mask_6108_hand_draw_numpy, OSSICLES_MESH_PATH_6108, gt_pose_6108), # error: 0.8516761480978112
        (lazy_fixture("app_full"), "6742", mask_6742_hand_draw_numpy, OSSICLES_MESH_PATH_6742, gt_pose_6742), # error: 2.415673998426594
        ]
)
def test_pnp_from_dataset(app, name, hand_draw_mask, ossicles_path, RT):

    # save the GT pose to .npy file
    np.save(DATA_DIR / f"{name}_gt_pose.npy", RT)
    
    mask = np.load(hand_draw_mask) / 255
    mask = np.where(mask != 0, 1, 0)
    
    plt.imshow(mask)
    plt.show()

    # expand the dimension
    ossicles_mask = np.expand_dims(mask, axis=-1)
        
    app.set_transformation_matrix(RT)
    app.load_meshes({'ossicles': ossicles_path})
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
            
    assert np.isclose(predicted_pose, RT, atol=7).all()