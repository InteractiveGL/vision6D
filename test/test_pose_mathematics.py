# Built-in Imports
import logging
import pathlib
import os

# Third-party Imports
import math
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

import vision6D
logger = logging.getLogger("vision6D")

# Constants
TEST_DATA_DIR = pathlib.Path(os.path.abspath(__file__)).parent / 'data'

def test_camera_rt():
    
    # Create 
    pl = pv.Plotter()
    # pl.camera.SetPosition((0.00000001, 0.00000001, 0.0000001))
    pl.camera.SetPosition(0,0,0)
    pl.camera.SetFocalPoint(0,0,1)
    
    # assert pl.camera.direction == (0,0,-1)
    
    # Then create a RT matrix based on the information
    t = np.array([0,0,-1])
    # r = R.from_rotvec((0,0,0)).as_matrix()
    r = np.eye(3)
    RT = np.vstack((np.hstack((r, t.reshape((-1,1)))), [0,0,0,1]))
    logger.debug(RT)
    
    # pl.camera.SetModelTransformMatrix(pv.vtkmatrix_from_array(RT))
    # pl.camera.SetFocalPoint((0,0,1))
    # pl.camera.SetViewUp(0,-1,0)
    
    # Adding cube
    cube = pv.Cube(center=(0,0,5))
    actor = pl.add_mesh(cube, color="red")
    pl.add_axes()
    pl.show()
    
    # Testing origin
    pt = np.array([0,0,1,1]).reshape((-1,1))
    t_pt = RT @ pt
    assert np.isclose(t_pt, np.array([0,0,0,1]).reshape((-1,1))).all()
    
    # Test other points
    pt = np.array([1,1,1,1]).reshape((-1,1))
    t_pt = RT @ pt
    assert np.isclose(t_pt, np.array([1,1,0,1]).reshape((-1,1))).all()
    
def test_others_code():

    w = 1024
    h = 768

    intrinsic = np.array([[665.10751011,   0.        , 511.5],
                          [  0.        , 665.10751011, 383.5],
                          [  0.        ,   0.        ,   1. ]])

    extrinsic = np.array([[ 0.95038793,  0.0954125 , -0.29607301, -1.84295291],
                          [-0.1222884 ,  0.98976322, -0.07358197, -1.2214318 ],
                          [ 0.28602154,  0.10613772,  0.95232687,  0.6428006 ],
                          [ 0.        ,  0.        ,  0.        ,  1.        ]])

    # extrinsic = np.array([[ 1,  0,  0, -1.84295291],
    #                       [ 0,  1,  0, -1.2214318 ],
    #                       [ 0,  0,  1,  0.6428006 ],
    #                       [ 0.        ,  0.        ,  0.        ,  1.        ]])

    # renderer
    p = pv.Plotter(off_screen=False, window_size=[w,h])


    #
    # load mesh or point cloud
    #
    mesh_filepath = TEST_DATA_DIR / 'fragment.ply'
    mesh = pv.read(str(mesh_filepath))
    p.add_mesh(mesh, rgb=True)


    #
    # intrinsics
    #

    cx = intrinsic[0,2]
    cy = intrinsic[1,2]
    f = intrinsic[0,0]

    # convert the principal point to window center (normalized coordinate system) and set it
    wcx = -2*(cx - float(w)/2) / w
    wcy =  2*(cy - float(h)/2) / h
    p.camera.SetWindowCenter(wcx, wcy)

    # convert the focal length to view angle and set it
    view_angle = 180 / math.pi * (2.0 * math.atan2(h/2.0, f))
    p.camera.SetViewAngle(view_angle)


    #
    # extrinsics
    #

    # apply the transform to scene objects
    p.camera.SetModelTransformMatrix(pv.vtkmatrix_from_array(extrinsic))

    # the camera can stay at the origin because we are transforming the scene objects
    p.camera.SetPosition(0, 0, 0)

    # look in the +Z direction of the camera coordinate system
    p.camera.SetFocalPoint(0, 0, 1)

    # the camera Y axis points down
    p.camera.SetViewUp(0,-1,0)


    #
    # near/far plane
    #

    # ensure the relevant range of depths are rendered
    # depth_min = 0.1
    # depth_max = 100
    # p.camera.SetClippingRange(depth_min, depth_max)
    # # depth_min, depth_max = p.camera.GetClippingRange()
    p.renderer.ResetCameraClippingRange()

    p.show()
    p.render()
    # p.store_image = True  # last_image and last_image_depth
    # p.close()


    # # get screen image
    # img = p.last_image

    # # get depth
    # # img = p.get_image_depth(fill_value=np.nan, reset_camera_clipping_range=False)
    # img = p.last_image_depth

    # plt.figure()
    # plt.imshow(img)
    # plt.colorbar(label='Distance to Camera')
    # plt.title('Depth image')
    # plt.xlabel('X Pixel')
    # plt.ylabel('Y Pixel')
    # plt.show()

def test_cube_on_others_code():
    
    z_offset = 5

    w = 1024
    h = 768

    intrinsic = np.array([[665.10751011,   0.        , 511.5],
                          [  0.        , 665.10751011, 383.5],
                          [  0.        ,   0.        ,   1. ]])

    # extrinsic = np.array([[ 0.95038793,  0.0954125 , -0.29607301, -1.84295291],
    #                       [-0.1222884 ,  0.98976322, -0.07358197, -1.2214318 ],
    #                       [ 0.28602154,  0.10613772,  0.95232687,  0.6428006 ],
    #                       [ 0.        ,  0.        ,  0.        ,  1.        ]])

    t = np.array([0,0,z_offset])
    r = R.from_rotvec((0,0,0.5)).as_matrix()
    # r = np.eye(3)
    RT = np.vstack((np.hstack((r, t.reshape((-1,1)))), [0,0,0,1]))
    # RT = np.array([[ 1,  0,  0,  0],
    #                       [ 0,  1,  0,  0],
    #                       [ 0,  0,  1,  0],
    #                       [ 0,  0,  0,  1]])

    # renderer
    p = pv.Plotter(off_screen=False, window_size=[w,h])


    #
    # load mesh or point cloud
    #
    # mesh_filepath = TEST_DATA_DIR / 'fragment.ply'
    cube = pv.Cube(center=(0,0,0))
    actor = p.add_mesh(cube, color="red")


    #
    # intrinsics
    #

    cx = intrinsic[0,2]
    cy = intrinsic[1,2]
    f = intrinsic[0,0]

    # convert the principal point to window center (normalized coordinate system) and set it
    wcx = -2*(cx - float(w)/2) / w
    wcy =  2*(cy - float(h)/2) / h
    p.camera.SetWindowCenter(wcx, wcy)

    # convert the focal length to view angle and set it
    view_angle = 180 / math.pi * (2.0 * math.atan2(h/2.0, f))
    p.camera.SetViewAngle(view_angle)


    #
    # extrinsics
    #

    # apply the transform to scene objects
    p.camera.SetModelTransformMatrix(pv.vtkmatrix_from_array(RT))

    # the camera can stay at the origin because we are transforming the scene objects
    p.camera.SetPosition(0, 0, 0)

    # look in the +Z direction of the camera coordinate system
    p.camera.SetFocalPoint(0, 0, 1)

    # the camera Y axis points down
    p.camera.SetViewUp(0,-1,0)


    #
    # near/far plane
    #

    # ensure the relevant range of depths are rendered
    # depth_min = 0.1
    # depth_max = 100
    # p.camera.SetClippingRange(depth_min, depth_max)
    # # depth_min, depth_max = p.camera.GetClippingRange()
    p.renderer.ResetCameraClippingRange()

    p.add_axes()
    p.show()
    p.render()
    # p.store_image = True  # last_image and last_image_depth
    # p.close()
    
    # Testing origin
    pt = np.array([0,0,z_offset,1]).reshape((-1,1))
    t_pt = np.linalg.inv(RT) @ pt
    assert np.isclose(t_pt, np.array([0,0,0,1]).reshape((-1,1))).all()
    
    # Test other points
    pt = np.array([1,1,z_offset,1]).reshape((-1,1))
    t_pt = np.linalg.inv(RT) @ pt
    assert np.isclose(t_pt, np.array([1,1,0,1]).reshape((-1,1))).all()
    
    # Test cube points
    pts = np.hstack((cube.points, np.ones(cube.points.shape[0]).reshape((-1,1))))
    t_pts = RT @ pts.T
    t_pts = t_pts[:3] / t_pts[3]
    
    assert (t_pts.T == (pts + np.array([0,0,z_offset,0]))[:,:3]).all()
    
    
    