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

# OSSICLES_MESH_PATH = DATA_DIR / "RL_20210304" / "5997_right_ossicles.mesh"
OSSICLES_MESH_PATH = DATA_DIR / "RL_20210304" / "5997_right_output_mesh_from_df.mesh"
FACIAL_NERVE_MESH_PATH = DATA_DIR / "RL_20210304" / "5997_right_facial_nerve.mesh"
CHORDA_MESH_PATH = DATA_DIR / "RL_20210304" / "5997_right_chorda.mesh"

OSSICLES_MESH_PATH_PLY = TEST_DATA_DIR / "ossicles_001_not_colored.ply"
OLD_OSSICLES_MESH_PATH = DATA_DIR / "RL_20210304" / "5997_right_output_mesh_from_df.mesh"
# OSSICLES_MESH_PATH = TEST_DATA_DIR / "ossicles_001_not_colored.ply"
# FACIAL_NERVE_MESH_PATH = TEST_DATA_DIR / "facial_nerve_001_not_colored.ply"
# CHORDA_MESH_PATH = TEST_DATA_DIR / "chorda_001_not_colored.ply"

OSSICLES_MESH_PATH_5997 = DATA_DIR / "RL_20210304" / "5997_right_ossicles.mesh"
OSSICLES_MESH_PATH_6088 = DATA_DIR / "RL_20210422" / "6088_right_ossicles.mesh"
OSSICLES_MESH_PATH_6108 = DATA_DIR / "RL_20210506" / "6108_right_ossicles.mesh"
OSSICLES_MESH_PATH_6742 = DATA_DIR / "RL_20211028" / "6742_left_ossicles.mesh"

mask_5997_hand_draw_numpy = DATA_DIR / "RL_20210304" / "mask_5997_hand_draw_numpy.npy"
mask_6088_hand_draw_numpy = DATA_DIR / "RL_20210422" / "mask_6088_hand_draw_numpy.npy"
mask_6108_hand_draw_numpy = DATA_DIR / "RL_20210506" / "mask_6108_hand_draw_numpy.npy"
mask_6742_hand_draw_numpy = DATA_DIR / "RL_20211028" / "mask_6742_hand_draw_numpy.npy"

# RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX = np.array([[ -0.15936675,  -0.30698457,  -0.93827647, -11.71427054],
#                                                       [  0.37909983,   0.85852426,  -0.34528161, -29.8785168 ],
#                                                       [  0.91152923,  -0.41072688,  -0.02044244, -90.48458992],
#                                                       [  0.,           0.,           0.,           1.        ]])

RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX = np.array([[ -0.14038217,  -0.3013128,   -0.94313491, -12.37503967],
                                        [  0.36747861,   0.8686707,   -0.33222081, -29.90434306],
                                        [  0.91937604,  -0.3932198,   -0.01121988, -90.78678434],
                                        [  0.,           0.,           0.,           1.        ]] ) #  GT pose

# RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX = np.array([[ -0.13050531,  -0.29699622,  -0.9459184 , -12.60770862],
#        [  0.3567857 ,   0.87609111,  -0.32429665, -29.3170283 ],
#        [  0.92502558,  -0.37981261,  -0.00837055, -90.38132077],
#        [  0.        ,   0.        ,   0.        ,   1.        ]])
                                        
RL_20210422_0_OSSICLES_TRANSFORMATION_MATRIX = np.array([[ -0.06365595,   0.02304574,  -1.0083763,  -21.32475993],
                                                        [  0.18774733,   0.99299512,   0.01084227, -25.96173491],
                                                        [  0.99101199,  -0.18664275,  -0.06682539, -87.674837  ],
                                                        [  0.,           0.,           0.,           1.        ]]) #  GT pose

# RL_20210422_0_OSSICLES_TRANSFORMATION_MATRIX = np.array([[ -0.06968065,   0.02342444,  -0.99729429, -21.03970757],
#                                                     [  0.18624669,   0.98245148,   0.01006281, -25.7154126 ],
#                                                     [  0.98002897,  -0.18504158,  -0.07282058, -90.28191254],
#                                                     [  0.        ,   0.        ,   0.        ,   1.        ]]) # Predicted pose from the pnp

RL_20210506_0_OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.28359947,   0.33634119,  -0.89802335, -31.3647034 ],
                                                        [  0.61564819,   0.65413142,   0.43941937, -16.56351629],
                                                        [  0.73522013,  -0.67748556,  -0.02155654, -12.46718709],
                                                        [  0.,           0.,           0.,           1.        ]]) #  GT pose

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

def test_compute_rigid_transformation():
    ply_data = pv.get_reader(OSSICLES_MESH_PATH_PLY).read()
    ply_data.points = ply_data.points.astype("double")
    mesh_data = pv.wrap(vis.utils.load_trimesh(OLD_OSSICLES_MESH_PATH))
    
    ply_vertices = ply_data.points
    mesh_vertices = mesh_data.points

    """
    ply_transfromed_vertices_pv = ply_data.transform(RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX)
    ply_transfromed_vertices = vis.utils.transform_vertices(ply_vertices, RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX)
    assert np.isclose(ply_transfromed_vertices_pv.points, ply_transfromed_vertices, atol=1e-10).all()

    mesh_transformed_vertices_pv = mesh_data.transform(RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX)
    mesh_transformed_vertices = vis.utils.transform_vertices(mesh_vertices, RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX)
    assert np.isclose(mesh_transformed_vertices_pv.points, mesh_transformed_vertices, atol=1e-10).all()

    rt = vis.utils.rigid_transform_3D(mesh_transformed_vertices, ply_transfromed_vertices)
    # rt = np.linalg.inv(rt)

    mesh_transformed = vis.utils.transform_vertices(mesh_transformed_vertices, rt)
    mesh_transformed_pv = mesh_transformed_vertices_pv.transform(rt)
    assert np.isclose(mesh_transformed_pv.points, mesh_transformed, atol=1e-10).all()

    # rt = vis.utils.rigid_transform_3D(ply_vertices, mesh_vertices) # input data shape need to be 3 by N
    # rt = np.linalg.inv(rt)
    gt_pose = RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX @ rt
    """

    rt = vis.utils.rigid_transform_3D(mesh_vertices, ply_vertices)
    gt_pose = rt @ RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX
    print(gt_pose)
    
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
    [(DATA_DIR / "RL_20210304" / "RL_20210304_0.png", DATA_DIR / "RL_20210304" / "5997_right_ossicles.mesh", DATA_DIR / "RL_20210304" / "5997_right_facial_nerve.mesh", DATA_DIR / "RL_20210304" / "5997_right_chorda.mesh", RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX),
     (DATA_DIR / "RL_20210422" / "RL_20210422_0.png", DATA_DIR / "RL_20210422" / "6088_right_ossicles.mesh", DATA_DIR / "RL_20210422" / "6088_right_facial_nerve.mesh", DATA_DIR / "RL_20210422" / "6088_right_chorda.mesh", RL_20210422_0_OSSICLES_TRANSFORMATION_MATRIX),
    (DATA_DIR / "RL_20210506" / "RL_20210506_0.png", DATA_DIR / "RL_20210506" / "6108_right_ossicles.mesh", DATA_DIR / "RL_20210506" / "6108_right_facial_nerve.mesh", DATA_DIR / "RL_20210506" / "6108_right_chorda.mesh", RL_20210506_0_OSSICLES_TRANSFORMATION_MATRIX),
    (DATA_DIR / "RL_20211028" / "RL_20211028_0.png", DATA_DIR / "RL_20211028" / "6742_left_ossicles.mesh", DATA_DIR / "RL_20211028" / "6742_left_facial_nerve.mesh", DATA_DIR / "RL_20211028" / "6742_left_chorda.mesh", RL_20211028_0_OSSICLES_TRANSFORMATION_MATRIX),
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
def test_pnp_with_masked_ossicles_surgical_microscope(app, name, hand_draw_mask, RT):
    
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
    app.load_meshes({'ossicles': OSSICLES_MESH_PATH})
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

@pytest.mark.parametrize(
    "app, name, hand_draw_mask, ossicles_path, RT",
    [
        (lazy_fixture("app_full"), "5997",  mask_5997_hand_draw_numpy, OSSICLES_MESH_PATH_5997, RL_20210304_0_OSSICLES_TRANSFORMATION_MATRIX), # error: 0.08919806077190386
        (lazy_fixture("app_full"), "6088", mask_6088_hand_draw_numpy, OSSICLES_MESH_PATH_6088, RL_20210422_0_OSSICLES_TRANSFORMATION_MATRIX), # error: 3.1873387451473523
        (lazy_fixture("app_full"), "6108", mask_6108_hand_draw_numpy, OSSICLES_MESH_PATH_6108, RL_20210506_0_OSSICLES_TRANSFORMATION_MATRIX), # error: 3.389128987491239
        (lazy_fixture("app_full"), "6742", mask_6742_hand_draw_numpy, OSSICLES_MESH_PATH_6742, RL_20211028_0_OSSICLES_TRANSFORMATION_MATRIX), # error: 0.827998127071814
        ]
)
def test_pnp_from_dataset(app, name, hand_draw_mask, ossicles_path, RT):
    
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
            
    assert np.isclose(predicted_pose, RT, atol=4).all()