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
import torchvision

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

OSSICLES_MESH_PATH_5997_right = DATA_DIR / "5997_right_ossicles.mesh"
OSSICLES_MESH_PATH_6088_right = DATA_DIR / "6088_right_ossicles.mesh"
OSSICLES_MESH_PATH_6108_right = DATA_DIR / "6108_right_ossicles.mesh"
OSSICLES_MESH_PATH_6742_left = DATA_DIR / "6742_left_ossicles.mesh"
OSSICLES_MESH_PATH_6742_right = DATA_DIR / "6742_right_ossicles.mesh"

mask_5997_hand_draw_numpy = DATA_DIR / "mask_5997_hand_draw_numpy.npy"
mask_6088_hand_draw_numpy = DATA_DIR / "mask_6088_hand_draw_numpy.npy"
mask_6108_hand_draw_numpy = DATA_DIR / "mask_6108_hand_draw_numpy.npy"
mask_6742_hand_draw_numpy = DATA_DIR / "mask_6742_hand_draw_numpy.npy"

gt_pose_5997 = np.array([[  -0.14498174,   -0.34676691,   -0.92667849,   29.87789505],
                        [   0.33413722,    0.86439266,   -0.3757361,   -13.14321823],
                        [   0.93130693,   -0.36411267,   -0.00945343, -119.95253896],
                        [   0.,            0.,            0.,            1.        ]]) #  GT pose
                                        
# gt_pose_60881 = np.array([[  0.2720979,   -0.10571448,  -0.96757074, -23.77536492],
                        # [  0.33708389,   0.95272971,  -0.00929905, -27.7404459 ],
                        # [  0.91309533,  -0.32021318,   0.29176419, -20.44275252],
                        # [  0.,           0.,           0.,           1.        ]]) #  GT pose


gt_pose_6088 = np.array([[  0.36049218,  -0.12347807,  -0.93605796, -24.3777202 ],
                        [  0.31229879,   0.96116227,  -0.00651795, -27.43646144],
                        [  0.89102231,  -0.28692541,   0.38099733, -19.1631882 ],
                        [  0.,           0.,           0.,           1.        ]])


gt_pose_6108 = np.array([[  0.18750646,   0.35092506,  -0.91743823, -29.81739935],
                        [  0.55230585,   0.73470634,   0.39390969, -19.10118172],
                        [  0.81228048,  -0.58056712,  -0.0560558,  227.40282413],
                        [  0.,           0.,           0.,           1.        ]]) #  GT pose

gt_pose_6742 = np.array([[  0.08142244,  -0.22290108,   0.97143477, -21.33779758],
                        [ -0.58239236,   0.78031899,   0.22786272,  -9.0573175 ],
                        [ -0.80881982,  -0.58430931,  -0.06628042, 457.93700994],
                        [  0.,           0.,           0.,           1.        ]]) #  GT pose

# gt_pose_6742 = np.eye(4)

# gt_pose_6742_mirror = np.array([[0.23328984,   0.39176946,  -0.88999581, -12.24053709],
#                                 [  0.40846391,   0.79110787,   0.45530822,  -0.72599635],
#                                 [  0.88245853,  -0.46974996,   0.02453373, 492.60459054],
#                                 [  0.,           0.,           0.,           1.        ]])

# size of the (1920, 1080)
@pytest.fixture
def app():
    return vis.App(register=True, scale=1)

# # TODO
# @pytest.fixture
# def meshpaths():
#     meshs = {}
#     meshs['OSSICLES_MESH_PATH_5997_right'] = OSSICLES_MESH_PATH_5997_right

#     return meshs
    
def test_load_image(app):
    image_source = np.array(Image.open(IMAGE_NUMPY_PATH))
    app.set_image_opacity(1)
    app.load_image(image_source, scale_factor=[0.01, 0.01, 1])
    app.set_reference("image")
    app.plot()
    
@pytest.mark.parametrize(
    "image_path, ossicles_path, facial_nerve_path, chorda_path, RT, mirror_objects, mirror_image",
    [(DATA_DIR / "5997_0.png", DATA_DIR / "5997_right_ossicles.mesh", DATA_DIR / "5997_right_facial_nerve.mesh", DATA_DIR / "5997_right_chorda.mesh", gt_pose_5997, False, False),
    (DATA_DIR / "6088_0.png", DATA_DIR / "6088_right_ossicles.mesh", DATA_DIR / "6088_right_facial_nerve.mesh", DATA_DIR / "6088_right_chorda.mesh", gt_pose_6088, False, False),
    (DATA_DIR / "6108_0.png", DATA_DIR / "6108_right_ossicles.mesh", DATA_DIR / "6108_right_facial_nerve.mesh", DATA_DIR / "6108_right_chorda.mesh", gt_pose_6108, False, False),
    (DATA_DIR / "6742_0.png", DATA_DIR / "original" / "6742_left_ossicles.mesh", DATA_DIR / "original" / "6742_left_facial_nerve.mesh", DATA_DIR / "original" / "6742_left_chorda.mesh", gt_pose_6742, True, False),
    (DATA_DIR / "6742_0.png", DATA_DIR / "6742_left_ossicles.mesh", DATA_DIR / "6742_left_facial_nerve.mesh", DATA_DIR / "6742_left_chorda.mesh", gt_pose_6742, False, False),
    (DATA_DIR / "6742_0.png", DATA_DIR / "6742_left_ossicles.mesh", DATA_DIR / "6742_left_facial_nerve.mesh", DATA_DIR / "6742_left_chorda.mesh", gt_pose_6742, True, True),
    (DATA_DIR / "6742_0.png", DATA_DIR / "6742_right_ossicles.mesh", DATA_DIR / "6742_right_facial_nerve.mesh", DATA_DIR / "6742_right_chorda.mesh", gt_pose_6742, True, False),
    ]
)  
def test_load_mesh_from_dataset(app, image_path, ossicles_path, facial_nerve_path, chorda_path, RT, mirror_objects, mirror_image):
    app.set_mirror_objects(mirror_objects)
    image_numpy = np.array(Image.open(image_path)) # (H, W, 3)
    if mirror_image:
        image_numpy = image_numpy[:, ::-1, ...]
    app.load_image(image_numpy)
    app.set_transformation_matrix(RT)
    app.load_meshes({'ossicles': ossicles_path, 'facial_nerve': facial_nerve_path, 'chorda': chorda_path})
    app.bind_meshes("ossicles", "g")
    app.bind_meshes("chorda", "h")
    app.bind_meshes("facial_nerve", "j")
    app.set_reference("ossicles")
    app.plot()

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
    "mesh_path, pose, mirror_objects",
    [(OSSICLES_MESH_PATH_5997_right, gt_pose_5997, False), 
    (OSSICLES_MESH_PATH_6088_right, gt_pose_6088, False),
    (OSSICLES_MESH_PATH_6108_right, gt_pose_6108, False),
    (OSSICLES_MESH_PATH_6742_right, gt_pose_6742, False),
    (OSSICLES_MESH_PATH_6742_left, gt_pose_6742, True),
    ]
)
def test_render_scene(app, mesh_path, pose, mirror_objects):
    app.set_register(True)
    app.set_mirror_objects(mirror_objects)
    app.set_transformation_matrix(pose)
    app.load_meshes({'ossicles': mesh_path})
    image_np = app.render_scene(render_image=False, render_objects=['ossicles'])
    plt.imshow(image_np)
    image = Image.fromarray(image_np)
    plt.imshow(image)
    plt.show()
    print("hhh")
    
def test_save_plot(app):
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
    vis.utils.save_image(image_np, TEST_DATA_DIR, f"image_plot.png")
    

def test_generate_image(app):
    # image_np = app.render_scene(render_image=True, image_source=IMAGE_JPG_PATH)
    image_numpy = np.array(Image.open(IMAGE_NUMPY_PATH)) # (H, W, 3)
    image_np = app.render_scene(render_image=True, image_source=image_numpy)
    plt.imshow(image_np); plt.show()
    vis.utils.save_image(image_np, TEST_DATA_DIR, f"image.png")


@pytest.mark.parametrize(
    "name, hand_draw_mask, ossicles_path, RT, resize, mirror_objects",
    [("5997",  mask_5997_hand_draw_numpy, OSSICLES_MESH_PATH_5997_right, gt_pose_5997, 1/5, False), # error: 0.8573262508172124 # if resize cv2: 0.5510600582101389 # if resize torch: 0.5943676548096519
    ("6088", mask_6088_hand_draw_numpy, OSSICLES_MESH_PATH_6088_right, gt_pose_6088, 1/5, False), # error: 5.398165257981464 # if resize cv2: 6.120078001305548 # if resize torch: 5.234686698024397
    ("6108", mask_6108_hand_draw_numpy, OSSICLES_MESH_PATH_6108_right, gt_pose_6108, 1/5, False), # error: 0.8516761480978112 # if resize cv2: 0.21774485476235367 # if resize torch: 49.322628634236146
    ("6742", mask_6742_hand_draw_numpy, OSSICLES_MESH_PATH_6742_left, gt_pose_6742, 1/5, False), # error: 2.415673998426594 # if resize cv2: 148.14798220849184 # if resize torch: 212.11247242207978
    ("6742", mask_6742_hand_draw_numpy, OSSICLES_MESH_PATH_6742_left, gt_pose_6742, 1/5, True), # error: 5.214560773437986 # if resize cv2: 230.26984657453482 # if resize torch: ...
    ]
)
def test_pnp_from_dataset(name, hand_draw_mask, ossicles_path, RT, resize, mirror_objects):

    ossicles_side = ossicles_path.stem.split("_")[1]
    if mirror_objects:
        if ossicles_side == "right":
            ossicles_side = "left"
        elif ossicles_side == "left":
            ossicles_side = "right"

    # save the GT pose to .npy file
    gt_pose_name = f"{name}_{ossicles_side}_gt_pose.npy"
    np.save(DATA_DIR / gt_pose_name, RT)
    app = vis.App(register=True, scale=1, mirror_objects=mirror_objects)

    mask = np.load(hand_draw_mask).astype("bool")
    if mirror_objects:
        mask = mask[..., ::-1]
    
    plt.imshow(mask)
    plt.show()

    # expand the dimension
    seg_mask = np.expand_dims(mask, axis=-1)
        
    app.set_transformation_matrix(RT)
    app.load_meshes({'ossicles': ossicles_path})
    app.plot()

    # Create rendering
    color_mask_whole = app.render_scene(render_image=False, render_objects=['ossicles'])
    # save the rendered whole image
    vis.utils.save_image(color_mask_whole, TEST_DATA_DIR, f"rendered_mask_whole_{name}.png")

    color_mask_binarized = vis.utils.color2binary_mask(color_mask_whole)
    binary_mask = color_mask_binarized * seg_mask

    color_mask = (color_mask_whole * seg_mask).astype(np.uint8)

    # save the rendered partial image
    vis.utils.save_image(color_mask, TEST_DATA_DIR, f"rendered_mask_partial_{name}.png")
    assert (binary_mask == vis.utils.color2binary_mask(color_mask)).all(), "render_binary_mask is not the same as converted render_color_mask"

    resize_width = int(1920 * resize)
    resize_height = int(1080 * resize)
    downscale_color_mask = cv2.resize(color_mask, (resize_width, resize_height), interpolation = cv2.INTER_AREA)
    downscale_binary_mask = cv2.resize(binary_mask, (resize_width, resize_height), interpolation = cv2.INTER_AREA)
    # numpy implementation
    color_mask = cv2.resize(downscale_color_mask, (1920, 1080), interpolation = cv2.INTER_AREA)
    binary_mask = cv2.resize(downscale_binary_mask, (1920, 1080), interpolation = cv2.INTER_AREA)
    # # torch implementation
    # trans = torchvision.transforms.Resize((1080, 1920))
    # color_mask = trans(torch.tensor(downscale_color_mask).permute(2,0,1))
    # color_mask = color_mask.permute(1,2,0).detach().cpu().numpy()
    # binary_mask = trans(torch.tensor(downscale_binary_mask).unsqueeze(-1).permute(2,0,1))
    # binary_mask = binary_mask.permute(1,2,0).squeeze().detach().cpu().numpy()
    # make sure the binary mask only contains 0 and 1
    binary_mask = np.where(binary_mask != 0, 1, 0)
    binary_mask_bool = binary_mask.astype('bool')
    assert (binary_mask == binary_mask_bool).all(), "binary mask should be the same as binary mask bool"

    plt.subplot(221)
    plt.imshow(color_mask_whole)
    plt.subplot(222)
    plt.imshow(seg_mask)
    plt.subplot(223)
    plt.imshow(binary_mask)
    plt.subplot(224)
    plt.imshow(color_mask)
    plt.show()
    
    # Create 2D-3D correspondences
    # pts2d, pts3d = vis.utils.create_2d_3d_pairs(color_mask, app, 'ossicles', binary_mask=binary_mask)
    pts2d, pts3d = vis.utils.create_2d_3d_pairs(color_mask, app, 'ossicles')
    
    logger.debug(f"The total points are {pts3d.shape[0]}")

    predicted_pose = vis.utils.solve_epnp_cv2(pts2d, pts3d, app.camera_intrinsics, app.camera.position)
            
    logger.debug(f"\ndifference from predicted pose and RT pose: {np.sum(np.abs(predicted_pose - RT))}")
            
    assert np.isclose(predicted_pose, RT, atol=7).all()