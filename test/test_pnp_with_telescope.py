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

CROPPED_MASK_PATH_NUMPY = DATA_DIR / "cropped_mask_numpy.npy"

OSSICLES_PATH = DATA_DIR / "ossicles_001_colored_not_centered.ply"
FACIAL_NERVE_PATH = DATA_DIR / "facial_nerve_001_colored_not_centered.ply"
CHORDA_PATH = DATA_DIR / "chorda_001_colored_not_centered.ply"

OSSICLES_PATH_NO_COLOR = DATA_DIR / "ossicles_001_not_colored.ply"
FACIAL_NERVE_PATH_NO_COLOR = DATA_DIR / "facial_nerve_001_not_colored.ply"
CHORDA_PATH_NO_COLOR = DATA_DIR / "chorda_001_not_colored.ply"

OSSICLES_MESH_PATH = DATA_DIR / "5997_right_output_mesh_from_df.mesh"
FACIAL_NERVE_MESH_PATH = DATA_DIR / "5997_right_facial_nerve.mesh"
CHORDA_MESH_PATH = DATA_DIR / "5997_right_chorda.mesh"

OSSICLES_TRANSFORMATION_MATRIX = np.array([[ -0.19337066,  -0.12923262,  -0.97257736,  -6.72829707],
                                        [  0.33208419,   0.92415657,  -0.18882458, -22.79122096],
                                        [  0.92321605,  -0.3594907,   -0.13578866, -12.93687721],
                                        [  0.,           0.,           0.,           1.,        ]])

def test_render_image():
    app = vis.App(register=True, 
                  width=960, 
                  height=540, 
                  cam_focal_length=2015, 
                  cam_position=(4.8, 2.7, -20), # z position Have to be greater than the object z translation
                  cam_focal_point=(4.8, 2.7, 0), 
                  cam_viewup=(0,-1,0))
    image_np = app.render_scene(IMAGE_PATH, [0.01, 0.01, 1], render_image=True)
    image = Image.fromarray(image_np)
    image.save(DATA_DIR / "image_rendered.png")

@pytest.mark.parametrize(
    "RT",
    [
        (np.array([[ -0.19337066,  -0.12923262,  -0.97257736,  -6.72829707],
                    [  0.33208419,   0.92415657,  -0.18882458, -22.79122096],
                    [  0.92321605,  -0.3594907,   -0.13578866, -12.93687721],
                    [  0.,           0.,           0.,           1.,        ]])),
        (np.array([[ -0.14574589,  -0.18814922,  -0.97126619,  -5.90455811],
                [  0.30022033,   0.92704254,  -0.22463276, -23.03902215],
                [  0.94266955,  -0.32433316,  -0.07862642, -12.92929744],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.23675579,  -0.22234119,  -0.94578597, -10.50147974],
            [  0.3887324,    0.91382535,  -0.11751746, -22.3507759 ],
            [  0.89041216,  -0.33983471,   0.30278467,  -5.30877871],
            [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.02027062,  -0.26227887,  -0.9647792,   -6.98852229],
            [  0.63058126,   0.75219371,  -0.19123779, -24.19561146],
            [  0.77585847,  -0.60449518,   0.18063558,  -0.56238297],
            [  0.,           0.,           0.,           1.        ]])),
        (np.array([[ -0.22586508,  -0.15418171,  -0.96187991,  -5.8362826 ],
                [  0.19062896,   0.96131056,  -0.19885323, -21.98409817],
                [  0.95532484,  -0.22827617,  -0.18773499, -16.84821795],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.05892563,  -0.232426,    -0.97082745,  -8.40139289],
                [  0.37967967,   0.90464739,  -0.19353666, -23.16207176],
                [  0.92323947,  -0.35719917,   0.14155434,  -8.22818824],
                [  0.,           0.,           0.,           1.        ]])),
    ]
)
def test_pnp_cropped_ossicles_masked_with_telescope(RT):
    
    app = vis.App(register=True, 
                  width=960, 
                  height=540, 
                  cam_focal_length=2015, 
                  cam_position=(4.8, 2.7, -20), # z position Have to be greater than the object z translation
                  cam_focal_point=(4.8, 2.7, 0), 
                  cam_viewup=(0,-1,0))
    
    # the obtained mask is a 1 channel image
    mask = np.load(CROPPED_MASK_PATH_NUMPY) / 255 # read image path: DATA_DIR / "ossicles_mask.png"
   
    # convert 1 channel to 3 channels for calculation
    ossicles_mask = np.stack((mask, mask, mask), axis=-1)
    
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
    plt.imshow(mask_render_masked*255)
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
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, iterationsCount=250, confidence=0.999, reprojectionError=1., flags=cv2.SOLVEPNP_ITERATIVE)
        
        # Get a rotation matrix
        predicted_pose = np.eye(4)
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + np.array(app.camera.position)
            logger.debug(len(inliers))
        
    assert np.isclose(predicted_pose, RT, atol=.4).all()