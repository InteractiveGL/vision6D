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

OSSICLES_PATH = DATA_DIR / "ossicles_001_colored_not_centered.ply"
FACIAL_NERVE_PATH = DATA_DIR / "facial_nerve_001_colored_not_centered.ply"
CHORDA_PATH = DATA_DIR / "chorda_001_colored_not_centered.ply"

OSSICLES_PATH_NO_COLOR = DATA_DIR / "ossicles_001_not_colored.ply"
FACIAL_NERVE_PATH_NO_COLOR = DATA_DIR / "facial_nerve_001_not_colored.ply"
CHORDA_PATH_NO_COLOR = DATA_DIR / "chorda_001_not_colored.ply"

OSSICLES_MESH_PATH = DATA_DIR / "5997_right_output_mesh_from_df.mesh"
FACIAL_NERVE_MESH_PATH = DATA_DIR / "5997_right_facial_nerve.mesh"
CHORDA_MESH_PATH = DATA_DIR / "5997_right_chorda.mesh"

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[-0.71779818,  -0.08115559,  -0.69150528,   5.89592362],
#                                             [0.02921941,   0.98879733,  -0.14637644, -18.43942796],
#                                             [0.69563784,  -0.12527412,  -0.7073856,  -22.87306696],
#                                             [0.,           0.,           0.,           1.        ]] )

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[ -0.14238535,  -0.19839697,  -0.96972422,  -4.84974046],
#  [0.14277906,   0.96534304,  -0.21846499, -20.72520213],
#  [0.97945932,  -0.16956253,  -0.10912377, -14.96613613],
#  [0.,           0.,           0.,           1.        ]])

# OSSICLES_TRANSFORMATION_MATRIX = np.array([[-0.56426324,  -0.11638116,  -0.81735086,   1.65956969],
#  [-0.11414693,   0.99150373,  -0.06237645, -15.39538323],
#  [0.81766587,   0.05810136,  -0.57275366, -24.83038287],
#  [0.,           0.,           0.,           1.        ]])


OSSICLES_TRANSFORMATION_MATRIX = np.array([[-0.2898125,   -0.18053846,  -0.93990137,  -2.68705409],
 [0.13168167,   0.96518633,  -0.22599845, -20.71186596],
 [0.94798137,  -0.18926495,  -0.25594945, -16.42911933],
 [0.,           0.,           0.,           1.        ]] )

# OSSICLES_TRANSFORMATION_MATRIX = np.eye(4)

@pytest.fixture
def app():
    return vis.App(True)
    
@pytest.fixture
def configured_app(app):
    app.load_image(IMAGE_PATH, [0.01, 0.01, 1])
    app.set_transformation_matrix(OSSICLES_TRANSFORMATION_MATRIX)
    app.load_meshes({'ossicles': OSSICLES_PATH_NO_COLOR, 'facial_nerve': FACIAL_NERVE_PATH_NO_COLOR, 'chorda': CHORDA_PATH_NO_COLOR})
    app.bind_meshes("ossicles", "g", ['facial_nerve', 'chorda'])
    app.bind_meshes("chorda", "h", ['ossicles', 'facial_nerve'])
    app.bind_meshes("facial_nerve", "j", ['ossicles', 'chorda'])
    app.set_reference("ossicles")
    return app

def test_create_app(app):
    assert isinstance(app, vis.App)

def test_load_image(app):
    # image = Image.open(IMAGE_PATH)
    # image = np.array(image)[::-1, :]
    # Image.fromarray(image).save(DATA_DIR / "image.jpg")
    app.image_opacity=1.0
    app.load_image(IMAGE_PATH, scale_factor=[0.01, 0.01, 1])
    app.set_reference("image")
    app.plot()

def test_load_mesh_from_ply(configured_app):
    configured_app.plot()
    
def test_load_mesh_from_polydata(app):
    # Set camera intrinsics
    app.set_camera_intrinsics(focal_length=2015, width=1920, height=1080)
    # Set camera extrinsics
    app.set_camera_extrinsics(position=(0, 0, -5), focal_point=(0, 0, 1), viewup=(0,-1,0))
    app.set_transformation_matrix(np.eye(4))
    app.load_meshes({'sephere': pv.Sphere(radius=1)})
    app.bind_meshes("sephere", "g", ['sephere'])
    app.plot()

def test_load_mesh_from_meshfile(app):
    app.load_image(IMAGE_PATH, [0.01, 0.01, 1])
    app.set_reference("ossicles")
    app.set_transformation_matrix(OSSICLES_TRANSFORMATION_MATRIX)
    app.load_meshes({'ossicles': OSSICLES_MESH_PATH, 'facial_nerve': FACIAL_NERVE_MESH_PATH, 'chorda': CHORDA_MESH_PATH})
    app.bind_meshes("ossicles", "g", ['facial_nerve', 'chorda'])
    app.bind_meshes("chorda", "h", ['ossicles', 'facial_nerve'])
    app.bind_meshes("facial_nerve", "j", ['ossicles', 'chorda'])
    app.plot()
    
def test_render_image(app):
    image_np = app.render_scene(IMAGE_PATH, [0.01, 0.01, 1], render_image=True)
    image = Image.fromarray(image_np)
    image.save(DATA_DIR / "image_rendered.png")
    
def test_render_ossicles(app):
    app.set_transformation_matrix(OSSICLES_TRANSFORMATION_MATRIX)
    app.load_meshes({'ossicles': OSSICLES_MESH_PATH, 'facial_nerve': FACIAL_NERVE_MESH_PATH, 'chorda': CHORDA_MESH_PATH})
    image_np = app.render_scene(BACKGROUND_PATH, [0.01, 0.01, 1], False, ['ossicles'])
    image = Image.fromarray(image_np)
    image.save(DATA_DIR / "ossicles_rendered.png")
    
def test_render_whole_scene(app):
    app.set_transformation_matrix(OSSICLES_TRANSFORMATION_MATRIX)
    app.load_meshes({'ossicles': OSSICLES_MESH_PATH, 'facial_nerve': FACIAL_NERVE_MESH_PATH, 'chorda': CHORDA_MESH_PATH})
    image_np = app.render_scene(BACKGROUND_PATH, [0.01, 0.01, 1], False, ['ossicles', 'facial_nerve', 'chorda'])
    image = Image.fromarray(image_np)
    image.save(DATA_DIR / "whole_scene_rendered.png")

# @pytest.mark.parametrize(
#     "_app, expected_RT", 
#     (
#         (lazy_fixture("app"), np.eye(4)),
#     )
# )
def test_pnp_with_cube(app):
    # Reference:
    # https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    # https://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf: Lecture 12: Camera Projection PPT
    
    # Set camera intrinsics
    app.set_camera_intrinsics(focal_length=2015, width=1920, height=1080)
    
    cam_position = (0, 0, -3)
    
    # Set camera extrinsics
    app.set_camera_extrinsics(position=cam_position, focal_point=(0, 0, 1), viewup=(0,-1,0))
    
    # Load a cube mesh
    cube = pv.Cube(center=(0,0,0))

    # # Create a RT transformation matrix manually
    # t = np.array([0,0,5])
    # r = R.from_rotvec((0,0.7,0)).as_matrix()
    # RT = np.vstack((np.hstack((r, t.reshape((-1,1)))), [0,0,0,1]))
    
    RT = np.array([[0.34344301, -0.77880413, -0.52489144,  0.        ],
                    [-0.18896486, -0.60475937,  0.77366556,  0.        ],
                    [-0.91996695, -0.16652398, -0.35486699,  5.        ],
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
    plt.imshow(mask_render*255)
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
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, iterationsCount=250, reprojectionError=1.)
        
        # Get a rotation matrix
        predicted_pose = np.eye(4)
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + cam_position
            logger.debug(len(inliers))
    
    assert np.isclose(predicted_pose, RT, atol=1e-2).all()
    
def test_pnp_with_sphere(app):
    # Reference:
    # https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    # https://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf: Lecture 12: Camera Projection PPT
    
    # Set camera intrinsics
    app.set_camera_intrinsics(focal_length=2015, width=1920, height=1080)
    
    cam_position = (0, 0, -3)
    
    # Set camera extrinsics
    app.set_camera_extrinsics(position=cam_position, focal_point=(0, 0, 1), viewup=(0, -1, 0))
    
    # Load a cube mesh
    sphere = pv.Sphere(radius=1)
    
    # Set a RT transformation matrix
    RT = np.array([[0.34344301, -0.77880413, -0.52489144,  0.        ],
                    [-0.18896486, -0.60475937,  0.77366556,  0.        ],
                    [-0.91996695, -0.16652398, -0.35486699,  5.        ],
                    [ 0.,          0.,          0.,          1.        ]])
    
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
    plt.imshow(mask_render*255)
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
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, iterationsCount=250, reprojectionError=1.)
        
        # Get a rotation matrix
        predicted_pose = np.eye(4)
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + cam_position
            logger.debug(len(inliers))
            
    assert np.isclose(predicted_pose, RT, atol=1e-1).all()
    
def test_pnp_with_ossicles(app):
    
    # Set camera intrinsics
    app.set_camera_intrinsics(focal_length=2015, width=1920, height=1080)
    
    cam_position = (9.6, 5.4, -20)
    
    # Set camera extrinsics
    app.set_camera_extrinsics(position=cam_position, focal_point=(9.6, 5.4, 0), viewup=(0, -1, 0))
    
    RT = OSSICLES_TRANSFORMATION_MATRIX
    
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
    plt.imshow(mask_render*255)
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
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, iterationsCount=250, reprojectionError=1.)
        
        # Get a rotation matrix
        predicted_pose = np.eye(4)
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + cam_position
            logger.debug(len(inliers))
            
    assert np.isclose(predicted_pose, RT, atol=1e-1).all()
    
def test_pnp_with_ossicles_masked(app):
    
    # the obtained mask is a 1 channel image
    # mask = (np.array(Image.open(MASK_PATH)) / 255) # mask = (np.array(Image.open(MASK_PATH)) / 255).astype('uint8') # read image path: DATA_DIR / "mask.png"
    mask = np.load(MASK_PATH_NUMPY) / 255 # read image path: DATA_DIR / "ossicles_mask.png"
    
    # convert 1 channel to 3 channels for calculation
    ossicles_mask = np.stack((mask, mask, mask), axis=-1)
    
    # Set camera intrinsics
    app.set_camera_intrinsics(focal_length=2015, width=1920, height=1080)
    
    cam_position = (9.6, 5.4, -20)
    
    # Set camera extrinsics
    app.set_camera_extrinsics(position=cam_position, focal_point=(9.6, 5.4, 0), viewup=(0, -1, 0))
    
    RT = OSSICLES_TRANSFORMATION_MATRIX
    
    app.set_transformation_matrix(RT)
    app.load_meshes({'ossicles': OSSICLES_PATH_NO_COLOR})
    app.plot()
    
    # Create rendering
    render_white_bg = app.render_scene(BACKGROUND_PATH, (0.01, 0.01, 1), render_image=False, render_objects=['ossicles'])
    render_black_bg = vis.utils.change_mask_bg(render_white_bg, [255, 255, 255], [0, 0, 0])
    mask_render = vis.utils.color2binary_mask(render_black_bg)
    mask_render_masked = mask_render * ossicles_mask
    render_masked_black_bg = render_black_bg * ossicles_mask  # render_masked_white_bg = render_white_bg * ossicles_mask
    
    plt.subplot(221)
    plt.imshow(render_black_bg)
    plt.subplot(222)
    plt.imshow(ossicles_mask)
    plt.subplot(223)
    plt.imshow(mask_render_masked*255)
    plt.subplot(224)
    plt.imshow(render_masked_black_bg.astype('uint8'))
    plt.show()
    
    # Create 2D-3D correspondences
    pts2d, pts3d = vis.utils.create_2d_3d_pairs(mask_render_masked, render_black_bg, app, 'ossicles')
    
    logger.debug(f"The total points are {pts3d.shape[0]}")

    pts2d = pts2d.astype('float32')
    pts3d = pts3d.astype('float32')
    camera_intrinsics = app.camera_intrinsics.astype('float32')
    
    if pts2d.shape[0] < 4:
        predicted_pose = np.eye(4)
        inliers = []
    else:
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, iterationsCount=250, reprojectionError=1.)
        
        # Get a rotation matrix
        predicted_pose = np.eye(4)
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + cam_position
            logger.debug(len(inliers)) # 50703
            
    assert np.isclose(predicted_pose, RT, atol=0.2).all()