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
import trimesh
import torchvision
import vision6D as vis
from scipy.spatial import distance_matrix

import logging
import PIL

logger = logging.getLogger("vision6D")
np.set_printoptions(suppress=True)

# size: (1920, 1080)
@pytest.fixture
def app():
    app = vis.App(off_screen=False,
                  nocs_color=True,
                  point_clouds=False)
    app.set_image_opacity(0.8)
    app.set_mesh_opacity(0.99)
    return app
    
def test_load_image(app):
    image_source = np.array(Image.open(vis.config.IMAGE_PATH_5997))
    app.set_image_opacity(1)
    app.load_image(image_source, scale_factor=[0.01, 0.01, 1])
    app.set_reference("image")
    app.plot()

@pytest.mark.parametrize(
    "image_path, ossicles_path, facial_nerve_path, chorda_path, gt_pose",
    [# right ossicles
    (vis.config.IMAGE_PATH_455, vis.config.OSSICLES_MESH_PATH_455_right, vis.config.FACIAL_NERVE_MESH_PATH_455_right, vis.config.CHORDA_MESH_PATH_455_right, vis.config.gt_pose_455_right),
    (vis.config.IMAGE_PATH_5997, vis.config.OSSICLES_MESH_PATH_5997_right, vis.config.FACIAL_NERVE_MESH_PATH_5997_right, vis.config.CHORDA_MESH_PATH_5997_right, vis.config.gt_pose_5997_right),
    (vis.config.IMAGE_PATH_6088, vis.config.OSSICLES_MESH_PATH_6088_right, vis.config.FACIAL_NERVE_MESH_PATH_6088_right, vis.config.CHORDA_MESH_PATH_6088_right, vis.config.gt_pose_6088_right),
    (vis.config.IMAGE_PATH_6108, vis.config.OSSICLES_MESH_PATH_6108_right, vis.config.FACIAL_NERVE_MESH_PATH_6108_right, vis.config.CHORDA_MESH_PATH_6108_right, vis.config.gt_pose_6108_right),
    (vis.config.IMAGE_PATH_632, vis.config.OSSICLES_MESH_PATH_632_right, vis.config.FACIAL_NERVE_MESH_PATH_632_right, vis.config.CHORDA_MESH_PATH_632_right, vis.config.gt_pose_632_right),
    (vis.config.IMAGE_PATH_6320, vis.config.OSSICLES_MESH_PATH_6320_right, vis.config.FACIAL_NERVE_MESH_PATH_6320_right, vis.config.CHORDA_MESH_PATH_6320_right, vis.config.gt_pose_6320_right),
    (vis.config.IMAGE_PATH_6329, vis.config.OSSICLES_MESH_PATH_6329_right, vis.config.FACIAL_NERVE_MESH_PATH_6329_right, vis.config.CHORDA_MESH_PATH_6329_right, vis.config.gt_pose_6329_right),
    (vis.config.IMAGE_PATH_6602, vis.config.OSSICLES_MESH_PATH_6602_right, vis.config.FACIAL_NERVE_MESH_PATH_6602_right, vis.config.CHORDA_MESH_PATH_6602_right, vis.config.gt_pose_6602_right),
    (vis.config.IMAGE_PATH_6751, vis.config.OSSICLES_MESH_PATH_6751_right, vis.config.FACIAL_NERVE_MESH_PATH_6751_right, vis.config.CHORDA_MESH_PATH_6751_right, vis.config.gt_pose_6751_right),
    # left ossicles
    (vis.config.IMAGE_PATH_6742, vis.config.OSSICLES_MESH_PATH_6742_left, vis.config.FACIAL_NERVE_MESH_PATH_6742_left, vis.config.CHORDA_MESH_PATH_6742_left, vis.config.gt_pose_6742_left),
    (vis.config.IMAGE_PATH_6087, vis.config.OSSICLES_MESH_PATH_6087_left, vis.config.FACIAL_NERVE_MESH_PATH_6087_left, vis.config.CHORDA_MESH_PATH_6087_left, vis.config.gt_pose_6087_left),
    ]
)  
def test_load_mesh(app, image_path, ossicles_path, facial_nerve_path, chorda_path, gt_pose):
    # save the GT pose to .npy file
    path = ossicles_path.stem.split('_')
    gt_pose_name = f"{path[0]}_{path[1]}_gt_pose.npy" # path[0] -> name; path[1] -> side
    np.save(vis.config.OP_DATA_DIR / "gt_poses" / gt_pose_name, gt_pose)

    image_numpy = np.array(Image.open(image_path)) # (H, W, 3)
    app.load_image(image_numpy)
    app.set_reference('ossicles')
    app.set_transformation_matrix(gt_pose)
    app.load_meshes({'ossicles': ossicles_path, 'facial_nerve': facial_nerve_path, 'chorda': chorda_path})
    app.bind_meshes("ossicles", "g")
    app.bind_meshes("chorda", "h")
    app.bind_meshes("facial_nerve", "j")
    app.plot()

    # check the clipping range
    print(app.pv_plotter.camera.clipping_range)

@pytest.mark.parametrize(
    "image_path, ossicles_path, facial_nerve_path, chorda_path, RT, mirror_objects, mirror_image",
    [# mirror left to right ossicles
    (vis.config.IMAGE_PATH_6742, vis.config.OSSICLES_MESH_PATH_6742_left, vis.config.FACIAL_NERVE_MESH_PATH_6742_left, vis.config.CHORDA_MESH_PATH_6742_left, np.eye(4), True, False),
    (vis.config.IMAGE_PATH_6742, vis.config.OSSICLES_MESH_PATH_6742_left, vis.config.FACIAL_NERVE_MESH_PATH_6742_left, vis.config.CHORDA_MESH_PATH_6742_left, vis.config.gt_pose_6742_left, True, True),
    (vis.config.IMAGE_PATH_6742, vis.config.OSSICLES_MESH_PATH_6742_left, vis.config.FACIAL_NERVE_MESH_PATH_6742_left, vis.config.CHORDA_MESH_PATH_6742_left, vis.config.gt_pose_6742_left, True, False),
    ]
)  
def test_load_mesh_mirror_ossicles(app, image_path, ossicles_path, facial_nerve_path, chorda_path, RT, mirror_objects, mirror_image):
    app.set_mirror_objects(mirror_objects)
    image_numpy = np.array(Image.open(image_path)) # (H, W, 3)
    if mirror_image:
        image_numpy = image_numpy[:, ::-1, ...]
    app.load_image(image_numpy)
    app.set_transformation_matrix(RT)
    app.load_meshes({'ossicles': ossicles_path})#, 'facial_nerve': facial_nerve_path, 'chorda': chorda_path})
    app.bind_meshes("ossicles", "g")
    app.bind_meshes("chorda", "h")
    app.bind_meshes("facial_nerve", "j")
    app.set_reference("ossicles")
    app.plot()

@pytest.mark.parametrize(
    "mesh_path, gt_pose, mirror_objects",
    [(vis.config.OSSICLES_MESH_PATH_455_right, vis.config.gt_pose_455_right, False),
    (vis.config.OSSICLES_MESH_PATH_5997_right, vis.config.gt_pose_5997_right, False), 
    (vis.config.OSSICLES_MESH_PATH_6088_right, vis.config.gt_pose_6088_right, False),
    (vis.config.OSSICLES_MESH_PATH_6108_right, vis.config.gt_pose_6108_right, False),
    (vis.config.OSSICLES_MESH_PATH_632_right, vis.config.gt_pose_632_right, False),
    (vis.config.OSSICLES_MESH_PATH_6320_right, vis.config.gt_pose_6320_right, False),
    (vis.config.OSSICLES_MESH_PATH_6329_right, vis.config.gt_pose_6329_right, False),
    (vis.config.OSSICLES_MESH_PATH_6602_right, vis.config.gt_pose_6602_right, False),
    (vis.config.OSSICLES_MESH_PATH_6751_right, vis.config.gt_pose_6751_right, False),
    
    (vis.config.OSSICLES_MESH_PATH_6742_left, vis.config.gt_pose_6742_left, False),
    (vis.config.OSSICLES_MESH_PATH_6742_left, vis.config.gt_pose_6742_left, True),
    ]
)
def test_scene_render(mesh_path, gt_pose, mirror_objects):
    app = vis.App(off_screen=True)
    app.set_mirror_objects(mirror_objects)
    app.set_transformation_matrix(gt_pose)
    app.load_meshes({'ossicles': mesh_path})
    image_np = app.plot()
    plt.imshow(image_np)
    plt.show()
    print("hhh")

@pytest.mark.parametrize(
    "mesh_path, gt_pose, mirror_objects",
    [(vis.config.OSSICLES_MESH_PATH_455_right, vis.config.gt_pose_455_right, False),
    (vis.config.OSSICLES_MESH_PATH_5997_right, vis.config.gt_pose_5997_right, False), 
    (vis.config.OSSICLES_MESH_PATH_6088_right, vis.config.gt_pose_6088_right, False),
    (vis.config.OSSICLES_MESH_PATH_6108_right, vis.config.gt_pose_6108_right, False),
    (vis.config.OSSICLES_MESH_PATH_632_right, vis.config.gt_pose_632_right, False),
    (vis.config.OSSICLES_MESH_PATH_6320_right, vis.config.gt_pose_6320_right, False),
    (vis.config.OSSICLES_MESH_PATH_6329_right, vis.config.gt_pose_6329_right, False),
    (vis.config.OSSICLES_MESH_PATH_6602_right, vis.config.gt_pose_6602_right, False),
    (vis.config.OSSICLES_MESH_PATH_6751_right, vis.config.gt_pose_6751_right, False),
    
    (vis.config.OSSICLES_MESH_PATH_6742_left, vis.config.gt_pose_6742_left, False),
    (vis.config.OSSICLES_MESH_PATH_6742_left, vis.config.gt_pose_6742_left, True),
    ]
) 
def test_get_depth_map(mesh_path, gt_pose, mirror_objects):
    app = vis.App(off_screen=True)
    app.set_mirror_objects(mirror_objects)
    app.set_transformation_matrix(gt_pose)
    app.load_meshes({'ossicles': mesh_path})
    _, depth_map = app.plot(return_depth_map=True)
        
    # show the depth map
    plt.figure()
    plt.imshow(depth_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth image')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()

    depth = depth_map[~np.isnan(depth_map)]

    z = app.cam_position - np.mean(depth)

    assert np.isclose(z, app.transformation_matrix[2,3], atol=2)

def test_plot_mesh_off_screen():
    app = vis.App(off_screen=True, nocs_color=True)
    app.set_transformation_matrix(np.eye(4))
    # image_numpy = np.array(Image.open(vis.config.IMAGE_PATH_5997)) # (H, W, 3)
    # app.load_image(image_numpy)
    app.load_meshes({'ossicles': vis.config.OSSICLES_MESH_PATH_5997_right, 'facial_nerve': vis.config.FACIAL_NERVE_MESH_PATH_5997_right, 'chorda': vis.config.CHORDA_MESH_PATH_5997_right})
    app.set_reference("ossicles")
    image_np = app.plot()
    plt.imshow(image_np)
    plt.show() 

def test_render_surgery_image(app):
    # image_numpy = np.array(Image.open(vis.config.IMAGE_PATH_5997)) # (H, W, 3)
    # image_np = app.render_scene(render_image=True, image_source=image_numpy)
    # plt.imshow(image_np); plt.show()
    app = vis.App(off_screen=True)
    app.set_image_opacity(1)
    image_numpy = np.array(Image.open(vis.config.IMAGE_PATH_5997)) # (H, W, 3)
    app.load_image(image_numpy)
    image_np = app.plot()
    plt.imshow(image_np)
    plt.show() 

@pytest.mark.parametrize(
    "ossicles_path",
    [# right ossicles
    (vis.config.OSSICLES_MESH_PATH_455_right),
    (vis.config.OSSICLES_MESH_PATH_5997_right),
    (vis.config.OSSICLES_MESH_PATH_6088_right),
    (vis.config.OSSICLES_MESH_PATH_6108_right),
    (vis.config.OSSICLES_MESH_PATH_632_right),
    (vis.config.OSSICLES_MESH_PATH_6320_right),
    (vis.config.OSSICLES_MESH_PATH_6329_right),
    (vis.config.OSSICLES_MESH_PATH_6602_right),
    (vis.config.OSSICLES_MESH_PATH_6751_right),
    # left ossicles
    (vis.config.OSSICLES_MESH_PATH_6742_left),
    (vis.config.OSSICLES_MESH_PATH_6087_left),
    ]
)
def test_convert_mesh2ply(ossicles_path):
    output_name = ossicles_path.stem + ".ply"
    output_path = ossicles_path.parent / output_name
    vis.utils.mesh2ply(ossicles_path, output_path)

@pytest.mark.parametrize(
    "ossicles_path, RT, resize, mirror_objects",
    [(vis.config.OSSICLES_MESH_PATH_455_right, vis.config.gt_pose_455_right, 1/10, False), # 0.06399427888260885
    (vis.config.OSSICLES_MESH_PATH_5997_right, vis.config.gt_pose_5997_right, 1/10, False), # 0.09308439281200796
    (vis.config.OSSICLES_MESH_PATH_6088_right, vis.config.gt_pose_6088_right, 1/10, False), # 4.928643628913529
    (vis.config.OSSICLES_MESH_PATH_6108_right, vis.config.gt_pose_6108_right, 1/10, False), # 0.43566387421926067
    (vis.config.OSSICLES_MESH_PATH_632_right, vis.config.gt_pose_632_right, 1/10, False), # 1.5448084781672122
    (vis.config.OSSICLES_MESH_PATH_6320_right, vis.config.gt_pose_6320_right, 1/10, False), # 0.1864101709362344 
    (vis.config.OSSICLES_MESH_PATH_6329_right, vis.config.gt_pose_6329_right, 1/10, False), # 0.5583152006946035
    (vis.config.OSSICLES_MESH_PATH_6602_right, vis.config.gt_pose_6602_right, 1/10, False), # 0.172686707386939 
    (vis.config.OSSICLES_MESH_PATH_6751_right, vis.config.gt_pose_6751_right, 1/10, False), # 0.13776459337835786

    # (vis.config.OSSICLES_MESH_PATH_6742_left, vis.config.gt_pose_6742_left, 1/10, False), # 
    # (vis.config.OSSICLES_MESH_PATH_6742_left, vis.config.gt_pose_6742_left, 1/10, True), # 
    ]
)
def test_pnp_from_dataset_with_whole_mask(ossicles_path, RT, resize, mirror_objects):

    app = vis.App(off_screen=True, nocs_color=True, mirror_objects=mirror_objects)

    app.set_transformation_matrix(RT)
    app.load_meshes({'ossicles': ossicles_path})

    # Create rendering
    color_mask_whole = app.plot()
    
    # generate the color mask based on the segmentation mask
    color_mask = (color_mask_whole).astype(np.uint8)
    
    # Downscale color_mask
    downscale_color_mask = cv2.resize(color_mask, (int(color_mask.shape[1] * resize), int(color_mask.shape[0] * resize)), interpolation=cv2.INTER_LINEAR)
            
    # Upscale color_mask
    color_mask = cv2.resize(downscale_color_mask, (int(downscale_color_mask.shape[1] / resize), int(downscale_color_mask.shape[0] / resize)), interpolation=cv2.INTER_LINEAR)
    
    plt.subplot(211)
    plt.imshow(color_mask_whole)
    plt.subplot(212)
    plt.imshow(color_mask)
    plt.show()
    
    # Create 2D-3D correspondences
    vertices = getattr(app, f'ossicles_vertices')
    pts3d, pts2d = vis.utils.create_2d_3d_pairs(color_mask, vertices) #, binary_mask=binary_mask)
    
    logger.debug(f"The total points are {pts3d.shape[0]}")

    predicted_pose = vis.utils.solve_epnp_cv2(pts2d, pts3d, app.camera_intrinsics, app.camera.position)

    if mirror_objects: predicted_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            
    logger.debug(f"\ndifference from predicted pose and RT pose: {np.sum(np.abs(predicted_pose - RT))}")

    assert np.isclose(predicted_pose, RT, atol=20).all()

@pytest.mark.parametrize(
    "seg_mask_path, ossicles_path, RT, resize, mirror_objects",
    [(vis.config.SEG_MASK_PATH_455, vis.config.OSSICLES_MESH_PATH_455_right, vis.config.gt_pose_455_right, 1/10, False), # 37.93270817944855
    (vis.config.SEG_MASK_PATH_5997, vis.config.OSSICLES_MESH_PATH_5997_right, vis.config.gt_pose_5997_right, 1/10, False), # 1.187697898790113
    (vis.config.SEG_MASK_PATH_6088, vis.config.OSSICLES_MESH_PATH_6088_right, vis.config.gt_pose_6088_right, 1/10, False), # 3.897070644774853
    (vis.config.SEG_MASK_PATH_6108, vis.config.OSSICLES_MESH_PATH_6108_right, vis.config.gt_pose_6108_right, 1/10, False), # 62.65367721516149
    (vis.config.SEG_MASK_PATH_632, vis.config.OSSICLES_MESH_PATH_632_right, vis.config.gt_pose_632_right, 1/10, False), # 44387.529716506484
    (vis.config.SEG_MASK_PATH_6320, vis.config.OSSICLES_MESH_PATH_6320_right, vis.config.gt_pose_6320_right, 1/10, False), # 1691.1802315404218
    (vis.config.SEG_MASK_PATH_6329, vis.config.OSSICLES_MESH_PATH_6329_right, vis.config.gt_pose_6329_right, 1/10, False), # 103.09055584267297
    (vis.config.SEG_MASK_PATH_6602, vis.config.OSSICLES_MESH_PATH_6602_right, vis.config.gt_pose_6602_right, 1/10, False), # 27.626377736343358
    (vis.config.SEG_MASK_PATH_6751, vis.config.OSSICLES_MESH_PATH_6751_right, vis.config.gt_pose_6751_right, 1/10, False), # 131.4366286484413
    ]
)
def test_pnp_from_dataset_with_seg_mask(seg_mask_path, ossicles_path, RT, resize, mirror_objects):

    app = vis.App(off_screen=True, nocs_color=True, mirror_objects=mirror_objects)

    # Use the hand segmented mask
    seg_mask = np.array(PIL.Image.open(seg_mask_path)).astype("bool")
    if mirror_objects: seg_mask = seg_mask[..., ::-1]
    
    # expand the dimension
    seg_mask = np.expand_dims(seg_mask, axis=-1)
        
    app.set_transformation_matrix(RT)
    app.load_meshes({'ossicles': ossicles_path})

    # Create rendering
    color_mask_whole = app.plot()
    
    # generate the color mask based on the segmentation mask
    color_mask = (color_mask_whole * seg_mask).astype(np.uint8)
    
    # Downscale color_mask
    downscale_color_mask = cv2.resize(color_mask, (int(color_mask.shape[1] * resize), int(color_mask.shape[0] * resize)), interpolation=cv2.INTER_LINEAR)
            
    # Upscale color_mask
    color_mask = cv2.resize(downscale_color_mask, (int(downscale_color_mask.shape[1] / resize), int(downscale_color_mask.shape[0] / resize)), interpolation=cv2.INTER_LINEAR)
    
    plt.subplot(311)
    plt.imshow(color_mask_whole)
    plt.subplot(312)
    plt.imshow(seg_mask)
    plt.subplot(313)
    plt.imshow(color_mask)
    plt.show()
    
    # Create 2D-3D correspondences
    vertices = getattr(app, f'ossicles_vertices')
    pts3d, pts2d = vis.utils.create_2d_3d_pairs(color_mask, vertices) #, binary_mask=binary_mask)
    
    logger.debug(f"The total points are {pts3d.shape[0]}")

    predicted_pose = vis.utils.solve_epnp_cv2(pts2d, pts3d, app.camera_intrinsics, app.camera.position)

    if mirror_objects: predicted_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            
    logger.debug(f"\ndifference from predicted pose and RT pose: {np.sum(np.abs(predicted_pose - RT))}")

    assert np.isclose(predicted_pose, RT, atol=150).all()