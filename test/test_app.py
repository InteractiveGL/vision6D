import logging
import pathlib
import os

import pytest
import pyvista as pv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from pytest_lazyfixture  import lazy_fixture

import vision6D as vis

logger = logging.getLogger("vision6D")

np.set_printoptions(suppress=True)

CWD = pathlib.Path(os.path.abspath(__file__)).parent
DATA_DIR = CWD / 'data'
IMAGE_PATH = DATA_DIR / "image.jpg"
BACKGROUND_PATH = DATA_DIR / "black_background.jpg"
# MASK_PATH = DATA_DIR / "mask.png"
# MASK_PATH = DATA_DIR / "ossicles_mask.png"
MASK_PATH = DATA_DIR / "segmented_mask_updated.png"
MASK_PATH_NUMPY = DATA_DIR / "segmented_mask_numpy.npy"
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

OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -2.28768117],
                                        [  0.61244989,   0.70950026,  -0.34858929, -25.39078897],
                                        [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
                                        [  0.,           0.,           0.,           1.        ]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  -0.19211785,    0.432653  ,    0.88085306,   22.72583494],
#        [   0.97645239,    0.17403346,    0.12748759,   -5.42031752],
#        [  -0.09814002,    0.88460372,   -0.45589998, -517.47343282],
#        [   0.        ,    0.        ,    0.        ,    1.        ]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[0.3380,  -0.5647,  -0.7529,  -9.9934],
#                                         [0.7189,   0.6712,  -0.1807, -22.4102],
#                                         [  0.6074,  -0.4803,   0.6328,  15.6896],
#                                         [0,   0, 0, 1]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -1.03017257],
#        [  0.61244989,   0.70950026,  -0.34858929, -25.88228735],
#        [  0.78560891,  -0.49725556,   0.3681787 ,   0.43013393],
#        [  0.        ,   0.        ,   0.        ,   1.        ]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -1.03017257],
#                 [  0.61244989,   0.70950026,  -0.34858929, -25.88228735],
#                 [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
#                 [  0.,           0.,           0.,           1.        ]])

# # sample to see the flipped camera
# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.28289925,  -0.95724774,   0.06037193,  22.89958349],
#                                         [  0.90608815,   0.24607375,  -0.34416852, -19.01705655],
#                                         [  0.31459859,   0.15206731,   0.93696492,  11.01578722],
#                                         [  0.        ,   0.        ,   0.        ,   1.        ]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.27405315,  -0.95999139,   0.0575449 ,  23.07934883],
#                                         [  0.90607829,   0.2376796 ,  -0.35004359, -18.89619621],
#                                         [  0.32236158,   0.14807074,   0.93496421,   4.59932613],
#                                         [  0.        ,   0.        ,   0.        ,   1.        ]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.13475468,   0.82180511,  -0.55360414, -20.35367606],
#                                     [  0.92092791,   0.10232293,   0.37606089,  -7.7183049 ],
#                                     [  0.36569516,  -0.56050548,  -0.74303478, -11.09524467],
#                                     [  0.,           0.,           0.,           1.        ]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.20626356,  -0.53811086,  -0.81724663,  -3.90666137],
#                                             [  0.68166679,   0.67821075,  -0.27451878, -25.23321459],
#                                             [  0.70198699,  -0.50046667,   0.50670246, 149.19832153],
#                                             [  0.        ,   0.        ,   0.        ,   1.        ]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.08654067,  -0.59102747,  -0.80199579,  -0.46290758],
#        [  0.55757269,   0.69586182,  -0.45264669, -26.76752479],
#        [  0.82560487,  -0.4079986 ,   0.38976112,  80.711312  ],
#        [  0.        ,   0.        ,   0.        ,   1.        ]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[  0.0873051 ,  -0.49934886,  -0.86199103,  -2.28334928],
#                                 [  0.61224162,   0.70948238,  -0.34899134, -25.39711749],
#                                 [  0.78583587,  -0.49727806,   0.36766359,   0.10044576],
#                                 [  0.        ,   0.        ,   0.        ,   1.        ]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[ -0.03921582,  -0.61266473,  -0.7893694 ,   1.68665061],
#                                         [  0.51137057,   0.6663846 ,  -0.54261561, -27.16172691],
#                                         [  0.85846506,  -0.42493939,   0.28716589,  81.78483878],
#                                         [  0.        ,   0.        ,   0.        ,   1.        ]])

@pytest.fixture
def app():
    return vis.App(True)
    
@pytest.fixture
def configured_app(app):
    app.load_image(IMAGE_PATH, [0.01, 0.01, 1])
    app.set_transformation_matrix(OSSICLES_TRANSFORMATION_MATRIX)
    app.load_meshes({'ossicles': OSSICLES_PATH_NO_COLOR, 'facial_nerve': FACIAL_NERVE_PATH_NO_COLOR, 'chorda': CHORDA_PATH_NO_COLOR})
    app.bind_meshes("ossicles", "g")
    app.bind_meshes("chorda", "h")
    app.bind_meshes("facial_nerve", "j")
    app.set_reference("ossicles")
    return app

def test_create_app(app):
    assert isinstance(app, vis.App)

def test_load_image(app):
    # image = Image.open(IMAGE_PATH)
    # image = np.array(image)[:, ::-1, :]
    # Image.fromarray(image).save(DATA_DIR / "image_flipped.jpg")
    app.image_opacity=1.0
    app.load_image(IMAGE_PATH, scale_factor=[0.01, 0.01, 1])
    app.set_reference("image")
    app.plot()
    
def test_load_mesh_from_ply(configured_app):
    configured_app.plot()
    
def test_load_mesh_from_polydata(app):
    # Set camera extrinsics
    app.set_camera_extrinsics(position=(0, 0, -300), focal_point=(0, 0, 1), viewup=(0,-1,0))
    app.set_transformation_matrix(np.eye(4))
    app.load_meshes({'sephere': pv.Sphere(radius=1)})
    app.plot()

def test_load_mesh_from_meshfile(app):
    app.load_image(IMAGE_PATH, [0.01, 0.01, 1])
    app.set_reference("ossicles")
    app.set_transformation_matrix(OSSICLES_TRANSFORMATION_MATRIX)
    app.load_meshes({'ossicles': OSSICLES_MESH_PATH, 'facial_nerve': FACIAL_NERVE_MESH_PATH, 'chorda': CHORDA_MESH_PATH})
    app.bind_meshes("ossicles", "g")
    app.bind_meshes("chorda", "h")
    app.bind_meshes("facial_nerve", "j")
    app.plot()
    
def test_render_ossicles(app):
    app.set_transformation_matrix(OSSICLES_TRANSFORMATION_MATRIX)
    app.load_meshes({'ossicles': OSSICLES_MESH_PATH, 'facial_nerve': FACIAL_NERVE_MESH_PATH, 'chorda': CHORDA_MESH_PATH})
    image_np = app.render_scene(BACKGROUND_PATH, [0.01, 0.01, 1], False, ['ossicles'])
    image = Image.fromarray(image_np)
    image.save(DATA_DIR / "ossicles_rendered.png")
    
def test_render_whole_scene(app):
    app.register = False
    app.set_transformation_matrix(OSSICLES_TRANSFORMATION_MATRIX)
    app.load_meshes({'ossicles': OSSICLES_MESH_PATH, 'facial_nerve': FACIAL_NERVE_MESH_PATH, 'chorda': CHORDA_MESH_PATH})
    image_np = app.render_scene(BACKGROUND_PATH, [0.01, 0.01, 1], False, ['ossicles', 'facial_nerve', 'chorda'])
    image = Image.fromarray(image_np)
    image.save(DATA_DIR / "whole_scene_rendered.png")

def test_pnp_with_cube(app):
    # Reference:
    # https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    # https://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf: Lecture 12: Camera Projection PPT
    
    cam_position = (0, 0, -200)
    
    # Set camera extrinsics
    app.set_camera_extrinsics(position=cam_position, focal_point=(0, 0, 1), viewup=(0,-1,0))
    
    # Load a cube mesh
    cube = pv.Cube(center=(0,0,0))

    # # Create a RT transformation matrix manually
    # t = np.array([0,0,5])
    # r = R.from_rotvec((0,0.7,0)).as_matrix()
    # RT = np.vstack((np.hstack((r, t.reshape((-1,1)))), [0,0,0,1]))
    
    # RT = np.eye(4)
    
    RT = np.array([[ 0.35755883, -0.70097575,  0.61707753,  0.        ],
                    [-0.12831925, -0.69136937, -0.71101516,  0.        ],
                    [ 0.92503289,  0.17504682, -0.33715391,  0.        ],
                    [ 0.,          0.,          0.,          1.        ]])
    
    app.set_transformation_matrix(RT)
    
    app.load_meshes({'cube': cube}) # Pass parameter of desired RT applied to
    
    app.plot()
    
    # Create rendering
    render_white_bg = app.render_scene(BACKGROUND_PATH, (0.01, 0.01, 1), render_image=False, render_objects=['cube'])
    render_black_bg = vis.utils.change_mask_bg(render_white_bg, [255, 255, 255], [0, 0, 0])
    mask_render = vis.utils.color2binary_mask(render_black_bg)
    
    plt.subplot(221)
    plt.imshow(render_white_bg)
    plt.subplot(222)
    plt.imshow(render_black_bg)
    plt.subplot(223)
    plt.imshow(mask_render)
    plt.show()
    
    pts2d, pts3d = vis.utils.create_2d_3d_pairs(mask_render, render_black_bg, app, 'cube')
    
    logger.debug(f"The total points are {pts3d.shape[0]}")

    pts2d = pts2d.astype('float32')
    pts3d = pts3d.astype('float32')
    camera_intrinsics = app.camera_intrinsics.astype('float32')
    
    if pts2d.shape[0] < 4:
        predicted_pose = np.eye(4)
        inliers = []
    else:
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, confidence=.99, flags=cv2.SOLVEPNP_EPNP)
        
        # Get a rotation matrix
        predicted_pose = np.eye(4)
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + cam_position
            logger.debug(len(inliers))
    
    assert np.isclose(predicted_pose, RT, atol=.1).all()
    
def test_pnp_with_sphere(app):
    # Reference:
    # https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    # https://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf: Lecture 12: Camera Projection PPT
    
    # set camera position
    cam_position = (0, 0, -300)
    
    # Set camera extrinsics
    app.set_camera_extrinsics(position=cam_position, focal_point=(0, 0, 1), viewup=(0, -1, 0))
    
    # Load a cube mesh
    sphere = pv.Sphere(radius=1)
    
    # Set a RT transformation matrix
    RT = np.array([[0.98405437,  0.02527457, -0.17606303, -0.41902711],
                    [0.09665508,  0.75496376,  0.64860428,  0.30374495],
                    [0.14931441, -0.65527927,  0.74048247,  0.50398105],
                    [0.,          0.,          0.,          1.,        ]])
    
    app.set_transformation_matrix(RT)
    
    app.load_meshes({'sphere': sphere}) # Pass parameter of desired RT applied to
    
    app.plot()
    
    # Create rendering
    render_white_bg = app.render_scene(BACKGROUND_PATH, (0.01, 0.01, 1), render_image=False, render_objects=['sphere'])
    render_black_bg = vis.utils.change_mask_bg(render_white_bg, [255, 255, 255], [0, 0, 0])
    mask_render = vis.utils.color2binary_mask(render_black_bg)
    
    plt.subplot(221)
    plt.imshow(render_white_bg)
    plt.subplot(222)
    plt.imshow(render_black_bg)
    plt.subplot(223)
    plt.imshow(mask_render)
    plt.show()
    
    # Create 2D-3D correspondences
    pts2d, pts3d = vis.utils.create_2d_3d_pairs(mask_render, render_black_bg, app, 'sphere')
    
    logger.debug(f"The total points are {pts3d.shape[0]}")

    pts2d = pts2d.astype('float32')
    pts3d = pts3d.astype('float32')
    camera_intrinsics = app.camera_intrinsics.astype('float32')
    
    if pts2d.shape[0] < 4:
        predicted_pose = np.eye(4)
        inliers = []
    else:
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, confidence=.99, flags=cv2.SOLVEPNP_EPNP)
        
        # Get a rotation matrix
        predicted_pose = np.eye(4)
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + cam_position
            logger.debug(len(inliers))
            
    assert np.isclose(predicted_pose, RT, atol=0.2).all()
    
@pytest.mark.parametrize(
    "RT",
    [
        (np.array([[  0.03292723,  -0.19892261,  -0.97946189 , -9.58437769],
                [  0.26335909 ,  0.94708607,  -0.18349376, -22.55842936],
                [  0.96413577,  -0.25190826,   0.083573,   -14.69781384],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.14574589,  -0.18814922,  -0.97126619,  -5.90455811],
                [  0.30022033,   0.92704254,  -0.22463276, -23.03902215],
                [  0.94266955,  -0.32433316,  -0.07862642, -12.92929744],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.43571207,   0.77372648,  -0.45989382, -24.09672039],
                    [  0.35232826,  -0.61678406,  -0.70387657,  -2.90416953],
                    [ -0.82826311,   0.14465392,  -0.54134597,  -3.38633483],
                    [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.14677223,  -0.10625233,  -0.98344718,  -8.07226061],
                [  0.08478254,   0.98920427,  -0.11952749, -19.49734302],
                [  0.98553023,  -0.10092246,  -0.13617937, -21.66010952],
                [  0.,           0.,           0.,           1.        ]]))
        
    ]
)
def test_pnp_with_ossicles_standard_len(RT):
    
    app = vis.App(register=True, 
                  width=1920, 
                  height=1080, 
                  cam_focal_length=2015, 
                  cam_position=(9.6, 5.4, -20), 
                  cam_focal_point=(9.6, 5.4, 0), 
                  cam_viewup=(0,-1,0))
    
    app.set_transformation_matrix(RT)
    app.load_meshes({'ossicles': OSSICLES_PATH_NO_COLOR})
    app.plot()
    
    # Create rendering
    render_white_bg = app.render_scene(BACKGROUND_PATH, (0.01, 0.01, 1), render_image=False, render_objects=['ossicles'])
    render_black_bg = vis.utils.change_mask_bg(render_white_bg, [255, 255, 255], [0, 0, 0])
    mask_render = vis.utils.color2binary_mask(render_black_bg)
    
    plt.subplot(221)
    plt.imshow(render_white_bg)
    plt.subplot(222)
    plt.imshow(render_black_bg)
    plt.subplot(223)
    plt.imshow(mask_render)
    plt.show()
    
    # Create 2D-3D correspondences
    pts2d, pts3d = vis.utils.create_2d_3d_pairs(mask_render, render_black_bg, app, 'ossicles')
    
    logger.debug(f"The total points are {pts3d.shape[0]}")

    pts2d = pts2d.astype('float32')
    pts3d = pts3d.astype('float32')
    camera_intrinsics = app.camera_intrinsics.astype('float32')
    
    if pts2d.shape[0] < 4:
        predicted_pose = np.eye(4)
        inliers = []
    else:
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        
        # Get a rotation matrix
        predicted_pose = np.eye(4)
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + app.camera.position
            logger.debug(len(inliers))
            
    assert np.isclose(predicted_pose, RT, atol=1).all()

@pytest.mark.parametrize(
    "RT",
    [
        (np.array([[  0.03292723,  -0.19892261,  -0.97946189 , -9.58437769],
                [  0.26335909 ,  0.94708607,  -0.18349376, -22.55842936],
                [  0.96413577,  -0.25190826,   0.083573,   -14.69781384],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.14574589,  -0.18814922,  -0.97126619,  -5.90455811],
                [  0.30022033,   0.92704254,  -0.22463276, -23.03902215],
                [  0.94266955,  -0.32433316,  -0.07862642, -12.92929744],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.43571207,   0.77372648,  -0.45989382, -24.09672039],
                    [  0.35232826,  -0.61678406,  -0.70387657,  -2.90416953],
                    [ -0.82826311,   0.14465392,  -0.54134597,  -3.38633483],
                    [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.14677223,  -0.10625233,  -0.98344718,  -8.07226061],
                [  0.08478254,   0.98920427,  -0.11952749, -19.49734302],
                [  0.98553023,  -0.10092246,  -0.13617937, -21.66010952],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.92722934,  -0.1715687,    0.33288124,   0.76798202],
            [ -0.25051527,  -0.94488288,   0.21080426,  27.04220591],
            [  0.27836637,  -0.27885573,  -0.91910372, -18.25491242],
            [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.81989509,  -0.45693302,   0.34494092,  29.99706976],
                    [  0.32592616,   0.12281218,   0.93738429,  11.94211439],
                    [ -0.47068478,   0.88098205,   0.04823333, -12.89403764],
                    [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.87676637,   0.46705484,   0.11463208,   8.80054401],
                [  0.41045146,   0.60251436,   0.68447502,  -3.08471196],
                [  0.25061989,   0.64717557,  -0.71996767, -32.34913297],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.60650966,  -0.05476751,   0.79318759,  20.45941968],
                [ -0.15432316,  -0.98676105,   0.04986971,  30.23673282],
                [  0.77995537,  -0.15265368,  -0.60693201, -22.86728747],
                [  0.,           0.,           0.,           1.        ]]))
    ]
)
def test_pnp_with_ossicles_surgical_microscope(RT):
    
    app = vis.App(register=True, 
                  width=1920, 
                  height=1080, 
                  cam_focal_length=5e+4, 
                  cam_position=(9.6, 5.4, -500), 
                  cam_focal_point=(9.6, 5.4, 0), 
                  cam_viewup=(0,-1,0))
    
    app.set_transformation_matrix(RT)
    app.load_meshes({'ossicles': OSSICLES_PATH_NO_COLOR})
    app.plot()
    
    # Create rendering
    render_white_bg = app.render_scene(BACKGROUND_PATH, (0.01, 0.01, 1), render_image=False, render_objects=['ossicles'])
    render_black_bg = vis.utils.change_mask_bg(render_white_bg, [255, 255, 255], [0, 0, 0])
    mask_render = vis.utils.color2binary_mask(render_black_bg)
    
    plt.subplot(221)
    plt.imshow(render_white_bg)
    plt.subplot(222)
    plt.imshow(render_black_bg)
    plt.subplot(223)
    plt.imshow(mask_render)
    plt.show()
    
    # Create 2D-3D correspondences
    pts2d, pts3d = vis.utils.create_2d_3d_pairs(mask_render, render_black_bg, app, 'ossicles')
    
    logger.debug(f"The total points are {pts3d.shape[0]}")

    pts2d = pts2d.astype('float32')
    pts3d = pts3d.astype('float32')
    camera_intrinsics = app.camera_intrinsics.astype('float32')
    
    if pts2d.shape[0] < 4:
        predicted_pose = np.eye(4)
        inliers = []
    else:
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        
        # Get a rotation matrix
        predicted_pose = np.eye(4)
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + app.camera.position
            logger.debug(len(inliers))
            
    assert np.isclose(predicted_pose, RT, atol=1).all()

# surgical microscope camera with focal length 50000  
@pytest.mark.parametrize(
    "RT",
    [
        (np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -2.28768117],
                    [  0.61244989,   0.70950026,  -0.34858929, -25.39078897],
                    [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
                    [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.04368987,  -0.36370424,  -0.93048935,  -4.26740943],
                [  0.43051434,   0.8336103,   -0.34605094, -25.10016632],
                [  0.9015257,   -0.41570794,   0.12015956,  -7.1369013 ],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.21403014,  -0.24968077,  -0.94437843,  -3.71577383],
                [  0.12515886,   0.95180355,  -0.28000937, -22.70262825],
                [  0.9687757,   -0.17812778,  -0.17246489, -17.75878025],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.03292723,  -0.19892261,  -0.97946189 , -9.58437769],
                [  0.26335909 ,  0.94708607,  -0.18349376, -22.55842936],
                [  0.96413577,  -0.25190826,   0.083573,   -14.69781384],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.14574589,  -0.18814922,  -0.97126619,  -5.90455811],
                [  0.30022033,   0.92704254,  -0.22463276, -23.03902215],
                [  0.94266955,  -0.32433316,  -0.07862642, -12.92929744],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.43571207,   0.77372648,  -0.45989382, -24.09672039],
                    [  0.35232826,  -0.61678406,  -0.70387657,  -2.90416953],
                    [ -0.82826311,   0.14465392,  -0.54134597,  -3.38633483],
                    [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.14677223,  -0.10625233,  -0.98344718,  -8.07226061],
                [  0.08478254,   0.98920427,  -0.11952749, -19.49734302],
                [  0.98553023,  -0.10092246,  -0.13617937, -21.66010952],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.92722934,  -0.1715687,    0.33288124,   0.76798202],
            [ -0.25051527,  -0.94488288,   0.21080426,  27.04220591],
            [  0.27836637,  -0.27885573,  -0.91910372, -18.25491242],
            [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.81989509,  -0.45693302,   0.34494092,  29.99706976],
                    [  0.32592616,   0.12281218,   0.93738429,  11.94211439],
                    [ -0.47068478,   0.88098205,   0.04823333, -12.89403764],
                    [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.87676637,   0.46705484,   0.11463208,   8.80054401],
                [  0.41045146,   0.60251436,   0.68447502,  -3.08471196],
                [  0.25061989,   0.64717557,  -0.71996767, -32.34913297],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -1.03017257],
                [  0.61244989,   0.70950026,  -0.34858929, -25.88228735],
                [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.16891985,  -0.13379315,  -0.97650678,  -6.21891939],
                    [  0.93714314,   0.28511639,  -0.20117495, -20.00981953],
                    [  0.30533393,  -0.94910908,   0.0772215,   11.38949457],
                    [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.29220856,   0.88228889,   0.36902639,  -7.69347165],
                [  0.75469319,   0.02427374,  -0.65562868, -23.58010659],
                [ -0.58741155,   0.47008203,  -0.65876442, -15.3083587 ],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.44872664,  -0.71230224,   0.53969429,  22.12477987],
                    [  0.89187583,   0.31870331,  -0.32091383, -20.74697582],
                    [  0.05658527,   0.62534288,   0.77829583,  -3.33870472],
                    [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.15904642,  -0.55171761,   0.81872579,  27.86719635],
                [  0.8514174,    0.49646224,   0.16915564, -16.26150026],
                [ -0.49979259,   0.6701738,    0.54870253,   0.20008692],
                [  0.,           0.,           0.,           1.        ]])),
        # (np.array([[  0.26894049,  -0.68035947,  -0.68174924,   3.35767679],
        #         [  0.95884839,   0.12225034,   0.25625106, -10.48357013],
        #         [ -0.09099875,  -0.72261044,   0.68523965,  23.3685089 ],
        #         [  0.,           0.,           0.,           1.        ]])), # not working, not enough points ~ 3000 points
        (np.array([[  0.26894049,  -0.68035947,  -0.68174924,   2.44142936],
                [  0.95884839,   0.12225034,   0.25625106, -10.72797506],
                [ -0.09099875,  -0.72261044,   0.68523965,  23.3685089 ],
                [  0.,           0.,           0.,           1.        ]])), # working, enough points ~30000 points
        (np.array([[ 0.26894049, -0.68035947, -0.68174924,  2.94090071],
                [ 0.95884839,  0.12225034,  0.25625106, -9.22736094],
                [-0.09099875, -0.72261044,  0.68523965, 23.3685089 ],
                [ 0.,          0.,          0.,          1.        ]])),
        (np.array([[ 0.35085479, -0.77589796, -0.52429314,  5.67395223],
                [ 0.89913432,  0.12269361,  0.42012348, -7.56528092],
                [-0.26164552, -0.61881228,  0.74068412, 24.86819684],
                [ 0.,          0.,          0.,          1.        ]])),
        # (np.array([[  0.27405315,  -0.95999139,   0.0575449,   23.07934883],
        #             [  0.90607829,   0.2376796,   -0.35004359, -18.89619621],
        #             [  0.32236158,   0.14807074,   0.93496421,   4.59932613],
        #             [  0.,           0.,           0.,           1.        ]])), # not very accurate
        (np.array([[  0.27405315,  -0.95999139,   0.0575449,   21.31368812],
                [  0.90607829,   0.2376796,   -0.35004359, -19.70032993],
                [  0.32236158,   0.14807074,   0.93496421,   4.69378537],
                [  0.,           0.,           0.,           1.        ]])), # accurate
        # (np.array([[  0.13475468,   0.82180511,  -0.55360414, -20.35367606],
        #         [  0.92092791,   0.10232293,   0.37606089,  -7.7183049 ],
        #         [  0.36569516,  -0.56050548,  -0.74303478, -11.09524467],
        #         [  0.,           0.,           0.,           1.        ]])), # not very accurate ~ 10000 points
        (np.array([[  0.13475468,   0.82180511,  -0.55360414, -21.0793958 ],
                [  0.92092791,   0.10232293,   0.37606089,  -7.7333164 ],
                [  0.36569516,  -0.56050548,  -0.74303478, -11.09524467],
                [  0.,           0.,           0.,           1.        ]])) # accurate ~40000 points
    ]
)
def test_pnp_with_ossicles_masked_surgical_microscope(app, RT):
    
    # surgical microscope intrinsic matrix
    app.set_camera_intrinsics(5e+4, 1920, 1080)
    app.set_camera_extrinsics((9.6, 5.4, -500), (9.6, 5.4, 0), (0,-1,0))
    
    # the obtained mask is a 1 channel image
    # mask = (np.array(Image.open(MASK_PATH)) / 255) # mask = (np.array(Image.open(MASK_PATH)) / 255).astype('uint8') # read image path: DATA_DIR / "mask.png"
    mask = np.load(MASK_PATH_NUMPY) / 255 # read image path: DATA_DIR / "ossicles_mask.png"
   
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
    vis.utils.save_image(render_black_bg, DATA_DIR, "rendered_mask_whole.png")
    mask_render = vis.utils.color2binary_mask(render_black_bg)
    mask_render_masked = mask_render * ossicles_mask
    render_masked_black_bg = (render_black_bg * ossicles_mask).astype(np.uint8)  # render_masked_white_bg = render_white_bg * ossicles_mask
    
    # # Save the rendered mask with black background
    # vis.utils.save_image(render_masked_black_bg, DATA_DIR, "rendered_mask.png")
    
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
    # pts2d, pts3d = vis.utils.create_2d_3d_pairs(mask_render_masked, render_masked_black_bg, app, 'ossicles', npts=30)
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
        
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + np.array(app.camera.position)
            logger.debug(len(inliers)) # 50703
            
    logger.debug(f"\ndifference from predicted pose and RT pose: {np.sum(np.abs(predicted_pose - RT))}")
            
    assert np.isclose(predicted_pose, RT, atol=2).all()
    
# standard camera with focal length 2015
@pytest.mark.parametrize(
    "RT",
    [
        (np.array([[ -0.13829142,  -0.12653892,  -0.9822746,   -6.48312688],
                    [  0.42244885,   0.88951583,  -0.17406479, -22.59560122],
                    [  0.89577477,  -0.43903245,  -0.06955619,  -8.81475365],
                    [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.21520382,  -0.16459101,  -0.96259915,  -4.40602881],
                    [  0.37154751,   0.8977676,   -0.23657087, -23.32077066],
                    [  0.90312776,  -0.40856228,  -0.13204963,  -9.19697377],
                    [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.51916417,  -0.02568808,  -0.85428841,  -3.69095739],
                    [  0.13335973,   0.98487039,  -0.11065937, -20.90143658],
                    [  0.84420598,  -0.17137806,  -0.50788366, -12.81557656],
                    [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.16693607,  -0.13527467,  -0.9766438,   -7.13516503],
                [  0.27834761,   0.94378566,  -0.17830097, -22.38437953],
                [  0.94586201,  -0.30161134,  -0.11989849,  -9.52593777],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.80482226,   0.03849138,   0.59226644,   4.50089248],
                [ -0.06887943,  -0.98509461,   0.15762053,  26.0791636 ],
                [  0.5895055,   -0.16765149,  -0.79017481, -22.86229215],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.19337066,  -0.12923262,  -0.97257736,  -5.38403053],
                [  0.33208419,   0.92415657,  -0.18882458, -22.44501157],
                [  0.92321605,  -0.3594907,   -0.13578866, -12.93687721],
                [  0.,           0.,           0.,           1.        ]]))
    ]
)
def test_pnp_with_ossicles_masked_standard_len(RT):
    
    # standard camera
    app = vis.App(register=True, 
                  width=1920, 
                  height=1080, 
                  cam_focal_length=2015, 
                  cam_position=(9.6, 5.4, -20), 
                  cam_focal_point=(9.6, 5.4, 0), 
                  cam_viewup=(0,-1,0))
    
    # the obtained mask is a 1 channel image
    # mask = (np.array(Image.open(MASK_PATH)) / 255) # mask = (np.array(Image.open(MASK_PATH)) / 255).astype('uint8') # read image path: DATA_DIR / "mask.png"
    mask = np.load(MASK_PATH_NUMPY) / 255 # read image path: DATA_DIR / "ossicles_mask.png"
   
    ossicles_mask = np.expand_dims(mask, axis=-1)
    
    app.set_transformation_matrix(RT)
    app.load_meshes({'ossicles': OSSICLES_PATH_NO_COLOR})
    app.plot()
    
    # Create rendering
    render_white_bg = app.render_scene(BACKGROUND_PATH, (0.01, 0.01, 1), render_image=False, render_objects=['ossicles'])
    render_black_bg = vis.utils.change_mask_bg(render_white_bg, [255, 255, 255], [0, 0, 0])
    mask_render = vis.utils.color2binary_mask(render_black_bg)
    mask_render_masked = mask_render * ossicles_mask
    render_masked_black_bg = (render_black_bg * ossicles_mask).astype(np.uint8)  # render_masked_white_bg = render_white_bg * ossicles_mask
    
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
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        
        # Get a rotation matrix
        predicted_pose = np.eye(4)
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + np.array(app.camera.position)
            logger.debug(len(inliers)) # 50703
            
    assert np.isclose(predicted_pose, RT, atol=.5).all()
    
@pytest.mark.parametrize(
    "RT",
    [
        (np.array([[ -0.40794128,   0.03516422,  -0.91233074,  -4.9792518 ],
                [  0.2583161,    0.96287463,  -0.0783916,  -20.40617284],
                [  0.87570354,  -0.2676489,   -0.4018798,  -14.71024412],
                [  0.,           0.,           0.,           1.        ]])),
    ]
)
def test_pnp_with_45_view_angle(RT):
    # Set up the camera parameters for 45 degree AOV
    app = vis.App(register=True, 
                  width=1920, 
                  height=1080, 
                  cam_focal_length=1303, 
                  cam_position=(9.6, 5.4, -13), # z position Have to be greater than the object z translation
                  cam_focal_point=(9.6, 5.4, 0), 
                  cam_viewup=(0,-1,0))
    
    # the obtained mask is a 1 channel image
    mask = np.load(MASK_PATH_NUMPY) / 255 # read image path: DATA_DIR / "ossicles_mask.png"
   
    ossicles_mask = np.expand_dims(mask, axis=-1)
    
    app.set_transformation_matrix(RT)
    app.load_meshes({'ossicles': OSSICLES_PATH_NO_COLOR})
    app.plot()
    
    # Create rendering
    render_white_bg = app.render_scene(BACKGROUND_PATH, (0.01, 0.01, 1), render_image=False, render_objects=['ossicles'])
    render_black_bg = vis.utils.change_mask_bg(render_white_bg, [255, 255, 255], [0, 0, 0])
    mask_render = vis.utils.color2binary_mask(render_black_bg)
    mask_render_masked = mask_render * ossicles_mask
    render_masked_black_bg = (render_black_bg * ossicles_mask).astype(np.uint8)  # render_masked_white_bg = render_white_bg * ossicles_mask
    
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
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        
        # Get a rotation matrix
        predicted_pose = np.eye(4)
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + np.array(app.camera.position)
            logger.debug(len(inliers))
        
    assert np.isclose(predicted_pose, RT, atol=.4).all()