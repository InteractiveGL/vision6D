# alpha = 6.835578651406617
# beta = 47.91692755829381
# gama = 172.76787223914218

# # rotation_matrix_extrinsic = np.array([
# #     [np.cos(beta)*np.cos(gama), np.sin(alpha)*np.sin(beta)*np.cos(gama)-np.cos(alpha)*np.sin(gama), np.cos(alpha)*np.sin(beta)*np.cos(gama)+np.sin(alpha)*np.sin(gama)],
# #     [np.cos(beta)*np.sin(gama), np.sin(alpha)*np.sin(beta)*np.sin(gama)+np.cos(alpha)*np.cos(gama), np.cos(alpha)*np.sin(beta)*np.sin(gama)-np.sin(alpha)*np.cos(gama)],
# #     [-np.sin(beta), np.sin(alpha)*np.cos(beta), np.cos(alpha)*np.cos(beta)]
# # ])

# rotation_matrix_intrinsic = np.array([
#     [np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gama)-np.sin(alpha)*np.cos(gama), np.cos(alpha)*np.sin(beta)*np.cos(gama)+np.sin(alpha)*np.sin(gama)],
#     [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gama)+np.cos(alpha)*np.cos(gama), np.sin(alpha)*np.sin(beta)*np.cos(gama)-np.cos(alpha)*np.sin(gama)],
#     [-np.sin(beta), np.cos(beta)*np.sin(gama), np.cos(beta)*np.cos(gama)]
# ])


# trans_vector = np.array(list(self.gt_position)).reshape((-1, 1))
# # self.transformation_matrix = np.vstack((np.hstack((rotation_matrix_intrinsic, trans_vector)), np.array([0, 0, 0, 1])))

def degree2matrix(self, r: list, t: list):
    rot = R.from_euler("xyz", r, degrees=True)
    rot = rot.as_matrix()
    
    # convert to euler angles
    rot_matrix = R.from_matrix(rot)
    euler = rot_matrix.as_euler('xyz', True)

    trans = np.array(t).reshape((-1, 1))
    matrix = np.vstack((np.hstack((rot, trans)), np.array([0, 0, 0, 1])))

    return matrix

  
# mesh = mesh.rotate_x(6.835578651406617, inplace=False)
# mesh = mesh.rotate_y(47.91692755829381, inplace=False)
# mesh = mesh.rotate_z(172.76787223914218, inplace=False)
# mesh = mesh.translate((2.5987030981091648, 31.039133701224685, 14.477777915423951), inplace=False)

 # # Load trimesh
# mesh_trimesh = self.load_trimesh(mesh_path)
# transformed_vertices = self.transform_vertices(self.transformation_matrix, mesh_trimesh.vertices)
# colors = self.color_mesh(transformed_vertices.T)
# mesh_trimesh.visual.vertex_colors = colors
# ply_file = trimesh.exchange.ply.export_ply(mesh_trimesh)
# with open("test/data/test.ply", "wb") as f:
#     f.write(ply_file)
# mesh = pv.read("test/data/test.ply")
# colors = self.color_mesh(mesh.points.T)

# self.gt_orientation = self.actors["ossicles"].orientation
# self.gt_position = self.actors["ossicles"].position
        
# actor.orientation = self.gt_orientation
# actor.position = self.gt_position



import numpy as np

def ralign(X,Y):
    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc =  X - np.tile(mx, (n, 1)).T
    Yc =  Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc*Xc, 0))
    sy = np.mean(np.sum(Yc*Yc, 0))

    Sxy = np.dot(Yc, Xc.T) / n

    U,D,V = np.linalg.svd(Sxy,full_matrices=True,compute_uv=True)
    V=V.T.copy()
    print(U,"\n\n",D,"\n\n",V)
    r = np.linalg.matrix_rank(Sxy)
    d = np.linalg.det(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if ( np.det(Sxy) < 0 ):
            S[m, m] = -1
        elif (r == m - 1):
            if (np.det(U) * np.det(V) < 0):
                S[m, m] = -1  
        else:
            R = np.eye(2)
            c = 1
            t = np.zeros(2)
            return R,c,t

    R = np.dot( np.dot(U, S ), V.T)

    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)

    return R,c,t

if __name__ == "__main__":

    # Run an example test
    # We have 3 points in 3D. Every point is a column vector of this matrix A
    A=np.array([[0.57215 ,  0.37512 ,  0.37551] ,[0.23318 ,  0.86846 ,  0.98642],[ 0.79969 ,  0.96778 ,  0.27493]])
    # Deep copy A to get B
    B=A.copy()
    # and sum a translation on z axis (3rd row) of 10 units
    B[2,:]=B[2,:]+10

    # Reconstruct the transformation with ralign.ralign
    R, c, t = ralign(A,B)
    print("Rotation matrix=\n",R,"\nScaling coefficient=",c,"\nTranslation vector=",t)
    
# temp = np.eye(4)
        
# temp1 = np.array([[-0.00000003, -0.99999997,  0.00023915, -0.23092513],
#                 [ 0.99999997, -0.00000009, -0.00023915,  0.22918788],
#                 [ 0.00023915,  0.00023915,  0.99999994,  2.50043322],
#                 [ 0.        ,  0.        ,  0.        ,  1.        ]])

# temp2 = np.array([[-0.99999926,  0.00003966,  0.00121823,  0.0612517 ],
#                 [-0.00004049, -0.99999976, -0.0006864 ,  0.4783421 ],
#                 [ 0.0012182 , -0.00068645,  0.99999902, -0.00021649],
#                 [ 0.        ,  0.        ,  0.        ,  1.        ]])

# tests
def test_different_origin():
    
    cube = pv.Cube()
    
    points = cube.points
    shifted_points = points - np.array([0, 0, 3])

    homogeneous_points = vis.utils.cartisian2homogeneous(points)
    
    RT = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,-3],
                   [0,0,0,1]])
    
    transformed_points = RT @ homogeneous_points.T
    
    transformed_points = vis.utils.homogeneous2cartisian(transformed_points)
    
    logger.debug((shifted_points == transformed_points).all())
    
def test_draw_axis(app):
    cube = pv.Cube()
    
    # Set camera intrinsics
    app.set_camera_intrinsics(focal_length=2015, width=1920, height=1080)
    
    # Set camera extrinsics
    app.set_camera_extrinsics(position=(0, 0, 4),focal_point=(0, 0, 0), viewup=(0, 1, 0))
    
    
    linex = pv.Line(pointa=(0, 0.0, 0.0), pointb=(1, 0.0, 0.0), resolution=1)
    liney =  pv.Line(pointa=(0.0, 0, 0.0), pointb=(0.0, 1, 0.0), resolution=1)
    linez =  pv.Line(pointa=(0.0, 0.0, 0), pointb=(0.0, 0.0, 1), resolution=1)
    
    # reference: https://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf: Lecture 12: Camera Projection PPT
    
    # app.load_image(IMAGE_PATH, [0.01, 0.01, 1])
    app.set_transformation_matrix(np.eye(4))
    app.load_meshes({'cube': cube}) # Pass parameter of desired RT applied to
    
    app.mesh_polydata['linex'] = linex
    app.mesh_polydata['liney'] = liney
    app.mesh_polydata['linez'] = linez
    
    mesh_linex = app.pv_plotter.add_mesh(linex, name='linex', color="red")
    mesh_liney = app.pv_plotter.add_mesh(liney, name='liney', color="green")
    mesh_linez = app.pv_plotter.add_mesh(linez, name='linez', color="blue")
    
    actor_linex, _ = app.pv_plotter.add_actor(mesh_linex, name='linex')
    actor_liney, _ = app.pv_plotter.add_actor(mesh_liney, name='liney')
    actor_linez, _ = app.pv_plotter.add_actor(mesh_linez, name='linez')
    
    app.mesh_actors['linex'] = actor_linex
    app.mesh_actors['liney'] = actor_liney
    app.mesh_actors['linez'] = actor_linez
    
    app.plot()

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
    
    # Create a plotter
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
    
# # read the mesh file
# if len(render_objects) == 1:
#     mesh = self.pv_render.add_mesh(self.mesh_polydata[f"{render_objects}"], rgb=True)
#     mesh.user_matrix = self.transformation_matrix
# else:
#     for _, mesh_data in self.mesh_polydata.items():
#         mesh = self.pv_render.add_mesh(mesh_data, rgb=True)
#         mesh.user_matrix = self.transformation_matrix


def event_change_color(self, *args):
    transformation_matrix = self.mesh_actors[self.reference].user_matrix
    container = self.mesh_actors.copy()

    for actor_name, actor in container.items():
        
        # Color the vertex
        transformed_points = utils.transform_vertices(transformation_matrix, self.mesh_polydata[actor_name].points)
        colors = utils.color_mesh(transformed_points.T)
        self.mesh_polydata[actor_name].point_data.set_scalars(colors)
        
        mesh = self.pv_plotter.add_mesh(self.mesh_polydata[actor_name], rgb=True, render=False, name=actor_name)
        mesh.user_matrix = transformation_matrix
        actor, _ = self.pv_plotter.add_actor(mesh, name=actor_name)
        
        # Save the new actor to a container
        self.mesh_actors[actor_name] = actor

    logger.debug("event_change_color callback complete")
    
# predicted_pose[:3, :3] = (cv2.Rodrigues(rotation_vector)[0].T @ view_matrix_r).T
# predicted_pose[:3, 3] = -(np.squeeze(translation_vector) - np.array(app.camera.position) + [0,0,10])

    
# view_matrix = pv.array_from_vtkmatrix(app.camera.GetViewTransformMatrix())
# view_matrix_r = np.array(view_matrix[:3,:3]) * [-1, 1, -1]


# # use standard camera with view angle 30 degree
# cam_focal_length:int=2015,
# cam_position: Tuple=(9.6, 5.4, -20), 
# # use standard camera with view angle 45 degree
# cam_focal_length:int=1303,
# cam_position: Tuple=(9.6, 5.4, -13), 


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
        (np.array([[ 0.35085479, -0.77589796, -0.52429314,  5.67395223],
                [ 0.89913432,  0.12269361,  0.42012348, -7.56528092],
                [-0.26164552, -0.61881228,  0.74068412, 24.86819684],
                [ 0.,          0.,          0.,          1.        ]])),
        (np.array([[  0.27405315,  -0.95999139,   0.0575449,   23.07934883],
                    [  0.90607829,   0.2376796,   -0.35004359, -18.89619621],
                    [  0.32236158,   0.14807074,   0.93496421,   4.59932613],
                    [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.13475468,   0.82180511,  -0.55360414, -20.35367606],
                [  0.92092791,   0.10232293,   0.37606089,  -7.7183049 ],
                [  0.36569516,  -0.56050548,  -0.74303478, -11.09524467],
                [  0.,           0.,           0.,           1.        ]]))
    ]
)

# # Set up the transformation for the scene object (not preferable, better to use user_matrix)
# self.camera.SetModelTransformMatrix(pv.vtkmatrix_from_array(matrix))
       
# test_create dataset code

@pytest.mark.parametrize(
    "RT",
    [
        (np.array([[  0.03292723,  -0.19892261,  -0.97946189 , -9.58437769],
                [  0.26335909 ,  0.94708607,  -0.18349376, -22.55842936],
                [  0.96413577,  -0.25190826,   0.083573,   -14.69781384],
                [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -1.73198707],
                    [  0.61244989,   0.70950026,  -0.34858929, -25.0914638 ],
                    [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
                    [  0.,           0.,           0.,           1.        ]]))
    ]
)
def test_pnp_with_ossicles_surgical_microscope_quarter_size(app_quarter, RT):
    
    app = app_quarter
   
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
        (np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -1.73198707],
                    [  0.61244989,   0.70950026,  -0.34858929, -25.0914638 ],
                    [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
                    [  0.,           0.,           0.,           1.        ]])),
        (np.array([[  0.05564547,  -0.42061121,  -0.90553289,  -3.67169989],
                [  0.5665481,    0.76009545,  -0.31824226, -24.86456215],
                [  0.82214769,  -0.49531921,   0.28059234,  -1.65629749],
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
        # (np.array([[  0.92722934,  -0.1715687,    0.33288124,   0.76798202],
        #     [ -0.25051527,  -0.94488288,   0.21080426,  27.04220591],
        #     [  0.27836637,  -0.27885573,  -0.91910372, -18.25491242],
        #     [  0.,           0.,           0.,           1.        ]])),
        # (np.array([[  0.29220856,   0.88228889,   0.36902639,  -7.69347165],
        #         [  0.75469319,   0.02427374,  -0.65562868, -23.58010659],
        #         [ -0.58741155,   0.47008203,  -0.65876442, -15.3083587 ],
        #         [  0.,           0.,           0.,           1.        ]])),
        # (np.array([[  0.15904642,  -0.55171761,   0.81872579,  27.86719635],
        #         [  0.8514174,    0.49646224,   0.16915564, -16.26150026],
        #         [ -0.49979259,   0.6701738,    0.54870253,   0.20008692],
        #         [  0.,           0.,           0.,           1.        ]])),
        # (np.array([[  0.26894049,  -0.68035947,  -0.68174924,   3.35767679],
        #         [  0.95884839,   0.12225034,   0.25625106, -10.48357013],
        #         [ -0.09099875,  -0.72261044,   0.68523965,  23.3685089 ],
        #         [  0.,           0.,           0.,           1.        ]])), # not working, not enough points ~ 3000 points
        # (np.array([[  0.26894049,  -0.68035947,  -0.68174924,   2.44142936],
        #         [  0.95884839,   0.12225034,   0.25625106, -10.72797506],
        #         [ -0.09099875,  -0.72261044,   0.68523965,  23.3685089 ],
        #         [  0.,           0.,           0.,           1.        ]])), # working, enough points ~30000 points
        # (np.array([[ 0.35085479, -0.77589796, -0.52429314,  5.67395223],
        #         [ 0.89913432,  0.12269361,  0.42012348, -7.56528092],
        #         [-0.26164552, -0.61881228,  0.74068412, 24.86819684],
        #         [ 0.,          0.,          0.,          1.        ]])),
        # (np.array([[  0.27405315,  -0.95999139,   0.0575449,   23.07934883],
        #             [  0.90607829,   0.2376796,   -0.35004359, -18.89619621],
        #             [  0.32236158,   0.14807074,   0.93496421,   4.59932613],
        #             [  0.,           0.,           0.,           1.        ]])), # not very accurate
        # (np.array([[  0.27405315,  -0.95999139,   0.0575449,   21.31368812],
        #         [  0.90607829,   0.2376796,   -0.35004359, -19.70032993],
        #         [  0.32236158,   0.14807074,   0.93496421,   4.69378537],
        #         [  0.,           0.,           0.,           1.        ]])), # accurate
        # (np.array([[  0.13475468,   0.82180511,  -0.55360414, -20.35367606],
        #         [  0.92092791,   0.10232293,   0.37606089,  -7.7183049 ],
        #         [  0.36569516,  -0.56050548,  -0.74303478, -11.09524467],
        #         [  0.,           0.,           0.,           1.        ]])), # not very accurate ~ 10000 points
        # (np.array([[  0.13475468,   0.82180511,  -0.55360414, -21.0793958 ],
        #         [  0.92092791,   0.10232293,   0.37606089,  -7.7333164 ],
        #         [  0.36569516,  -0.56050548,  -0.74303478, -11.09524467],
        #         [  0.,           0.,           0.,           1.        ]])) # accurate ~40000 points
    ])
def test_pnp_with_masked_ossicles_surgical_microscope_smallest_size(app_smallest, RT):
    app = app_smallest
    mask = np.load(MASK_PATH_NUMPY_SMALLEST) / 255 # read image path: DATA_DIR / "ossicles_mask.png"
   
    mask = np.where(mask != 0, 1, 0)
   
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
    # save the rendered whole image
    vis.utils.save_image(render_black_bg, DATA_DIR, "rendered_mask_whole_smallest.png")
    mask_render = vis.utils.color2binary_mask(render_black_bg)
    mask_render_masked = mask_render * ossicles_mask
    render_masked_black_bg = (render_black_bg * ossicles_mask).astype(np.uint8)  # render_masked_white_bg = render_white_bg * ossicles_mask
    # save the rendered partial image
    vis.utils.save_image(render_masked_black_bg, DATA_DIR, "rendered_mask_partial_smallest.png")
    
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
        
        # Use EPNP
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
            
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + np.array(app.camera.position)
            logger.debug(len(inliers)) # 50703
            
    logger.debug(f"\ndifference from predicted pose and RT pose: {np.sum(np.abs(predicted_pose - RT))}")
            
    assert np.isclose(predicted_pose, RT, atol=20).all()

# (lazy_fixture("app_full"), np.array([[  0.37177922,   0.8300953,   -0.41559835, -24.30279375],
#             [  0.51533184,  -0.5569185,   -0.65136388,  -4.5669351],
#             [ -0.7721485,    0.0279925,   -0.63482527,  -3.57181275],
#             [  0.,           0.,           0.,           1.,        ]])),
# (lazy_fixture("app_full"),  np.array([[  0.08788493,  -0.49934587,  -0.86193385,  -2.28768117],
#             [  0.61244989,   0.70950026,  -0.34858929, -25.39078897],
#             [  0.78560891,  -0.49725556,   0.3681787,    0.43013393],
#             [  0.,           0.,           0.,           1.        ]])),
# (lazy_fixture("app_full"),   np.array([[  0.08725841,  -0.49920268,  -0.86208042,  -1.773618  ],
#                         [  0.61232186,   0.7094788 ,  -0.34885781, -25.13447245],
#                         [  0.78577854,  -0.49742991,   0.3675807 ,   2.70771307],
#                         [  0.        ,   0.        ,   0.        ,   1.        ]]))
# (lazy_fixture("app_full"),   np.array([[  0.08771557,  -0.49943043,  -0.8619021 ,  -1.75110001],
#                         [  0.61228039,   0.70952996,  -0.34882654, -25.11469416],
#                         [  0.78575995,  -0.49712824,   0.36802828,   1.49357594],
#                         [  0.        ,   0.        ,   0.        ,   1.        ]]))

# # test the trackball actor style
# import pyvista as pv
# plotter = pv.Plotter()
# _ = plotter.add_mesh(pv.Cube(center=(1, 0, 0)))
# _ = plotter.add_mesh(pv.Cube(center=(0, 1, 0)))
# plotter.show_axes()
# plotter.enable_trackball_actor_style()
# plotter.show()  


# generate white image
# self.pv_render.add_mesh(background, rgb=True, opacity=1, name="image")
# generate grey image
# self.pv_render.add_mesh(background, rgb=True, opacity=0.5, name="image")

# def change_mask_bg(image, original_values, new_values):
    
#     new_image_bg = copy.deepcopy(image)
    
#     new_image_bg[np.where((new_image_bg[...,0] == original_values[0]) & (new_image_bg[...,1] == original_values[1]) & (new_image_bg[...,2] == original_values[2]))] = new_values

#     return new_image_bg
# render_black_bg = vis.utils.change_mask_bg(render_white_bg, [255, 255, 255], [0, 0, 0])

def meshwrite(output_filename, mesh):

    fid = open('test/data/nii_001/5997_right_output_mesh_from_df.mesh', 'rb')
    ossicle_mesh = meshread(fid, linesread=False, meshread2=False)
    # mesh.sz = np.array([1,1,1])

    # Unify the data dtypes to be the same as the the ones in the original ossicle_mesh file
    mesh.vertices = mesh.vertices.astype(str(ossicle_mesh.vertices.dtype))
    mesh.triangles = mesh.triangles.astype(str(ossicle_mesh.triangles.dtype))

    with open(output_filename, "wb") as f:
        f.write(mesh.id.T)
        f.write(mesh.numverts.T)
        f.write(mesh.numtris.T)
        f.write(np.int32(-1).T)
        f.write(mesh.orient.T)
        f.write(mesh.dim.T)
        f.write(mesh.sz.T)
        f.write(mesh.color.T)
        # ndarray need to be C-continuous!
        f.write(mesh.vertices.T.tobytes(order='C'))
        f.write(mesh.triangles.T.tobytes(order='C'))
        if hasattr(mesh, "opacity"):
            f.write(mesh.opacity.T)
            if hasattr(mesh, "colormap"):
                f.write(mesh.colormap.numcols.T)
                f.write(mesh.colormap.numverts.T)
                f.write(mesh.colormap.cols.T)
                f.write(mesh.colormap.vertexindexes.T.tobytes(order='C'))

def save_obj(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)
def convert2ply(obj: trimesh.Trimesh, filename):
    ply_file = trimesh.exchange.ply.export_ply(obj)
    with open(filename, "wb") as f:
        f.write(ply_file)


def center_mesh(mesh):
    centered_mesh = copy.deepcopy(mesh)
    centroid = np.mean(mesh.vertices, axis=1)
    centered_mesh.vertices = mesh.vertices - np.expand_dims(centroid, axis=1)
    return centered_mesh

def create_black_bg():
    image = Image.new('RGB', (1920, 1080), color=0)
    image.save("test/data/black_background.jpg")

def compare_two_images():
    # They are different
    image1 = np.array(Image.open("image.png"))
    image2 = np.array(Image.open("test/data/RL_20210304_0.jpg"))
    print("hhh")

def show_plot(frame, plot, image_white_bg, image_black_bg):
    extent = 0, plot.shape[1], plot.shape[0], 0
    plt.subplot(221)
    plt.imshow(frame, alpha=1, extent=extent, origin="upper")

    plt.subplot(222)
    im1 = plt.imshow(plot, extent=extent, origin="upper")
    im2 = plt.imshow(image_white_bg, alpha=0.5, extent=extent, origin="upper")
    
    plt.subplot(223)
    plt.imshow(image_white_bg, alpha=1, extent=extent, origin="upper")
    
    plt.subplot(224)
    plt.imshow(image_black_bg, alpha=1, extent=extent, origin="upper")
    plt.show()
    
    plt.close()
    
    print('hhh')

    
def count_white_black_pixels(image_grey):
    sought = [0,0,0]
    black  = np.count_nonzero(np.all(image_grey==sought,axis=2))
    print(f"black: {black}")
    
    sought = [255,255,255]
    white  = np.count_nonzero(np.all(image_grey==sought,axis=2))
    print(f"white: {white}")

def check_pixel_in_image(image, pixel_value):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    pixels = [i for i in image.getdata()]
    assert not pixel_value in pixels, f"{pixel_value} in pixels"


def load_mesh_color(mesh):
    colors = copy.deepcopy(mesh.vertices)
    colors[0] = (mesh.vertices[0] - np.min(mesh.vertices[0])) / (np.max(mesh.vertices[0]) - np.min(mesh.vertices[0])) - 0.5
    colors[1] = (mesh.vertices[1] - np.min(mesh.vertices[1])) / (np.max(mesh.vertices[1]) - np.min(mesh.vertices[1])) - 0.5
    colors[2] = (mesh.vertices[2] - np.min(mesh.vertices[2])) / (np.max(mesh.vertices[2]) - np.min(mesh.vertices[2])) - 0.5
    colors = colors.T + np.array([0.5, 0.5, 0.5])

    return colors
    
    
def cartisian2homogeneous(vertices):
    return np.hstack((vertices, np.ones(vertices.shape[0]).reshape((-1,1))))

def homogeneous2cartisian(homo_vertices):
    return (homo_vertices[:3] / homo_vertices[3]).T


def event_reset_image(self, *args):
    self.image_actors["image"] = self.image_actors["image-origin"].copy() # have to use deepcopy to prevent change self.image_actors["image-origin"] content
    self.pv_plotter.add_actor(self.image_actors["image"], name="image")
    logger.debug("reset_image_position callback complete")


self.pv_plotter.add_key_event('d', self.event_reset_image)

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

def color2binary_mask(color_mask):
    # binary_mask = copy.deepcopy(color_mask)
    binary_mask = np.zeros(color_mask[...,:1].shape)
    
    black_pixels_mask = np.all(color_mask == [0, 0, 0], axis=-1)
    non_black_pixels_mask = ~black_pixels_mask
    # non_black_pixels_mask = np.any(color_mask != [0, 0, 0], axis=-1)  

    binary_mask[black_pixels_mask] = [0]
    binary_mask[non_black_pixels_mask] = [1]
    
    return binary_mask

if isinstance(image_source, pathlib.Path):
                image = pv.get_reader(image_source).read()
                image = image.scale(scale_factor, inplace=False)
                
            elif isinstance(image_source, np.ndarray):

    
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

# # mirror the objects
# if self.mirror: 
#     mesh_data_reflect = mesh_data.reflect((1, 0, 0)) # pyvista implementation

#     mesh_data_vertices = mesh_data.points * np.array((-1, 1, 1))

#     # mirror the mesh along the x axis
#     mirror_x = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#     mesh_data.points = vis.utils.transform_vertices(mesh_data.points, mirror_x)

#     assert (mesh_data.points == mesh_data_reflect.points).all() and (mesh_data.points == mesh_data_vertices).all(), "mesh_data.points should equal to mesh_data_reflect!"

# elif isinstance(mesh_source, pv.PolyData):
#                 mesh_data = mesh_source

if '.ply' in str(mesh_source):
    mesh_data = pv.get_reader(mesh_source).read()
    # Convert the data type from float32 to float64 to match with load_trimesh
    mesh_data.points = mesh_data.points.astype("double")
elif '.mesh' in str(mesh_source): # .mesh obj data

MASK_PATH_NUMPY_FULL = TEST_DATA_DIR / "segmented_mask_numpy.npy"
MASK_PATH_NUMPY_QUARTER = TEST_DATA_DIR / "quarter_image_mask_numpy.npy"
MASK_PATH_NUMPY_SMALLEST = TEST_DATA_DIR / "smallest_image_mask_numpy.npy"
STANDARD_LENS_MASK_PATH_NUMPY = TEST_DATA_DIR / "test1.npy"

"""
if self.mirror_objects: 
    center = np.mean(mesh_data.points, axis=0)
    mesh_data = mesh_data.reflect((1, 0, 0), point = center) # mirror the object based on the center point
    mesh_name = mesh_name + '_reflect'
    self.mesh_polydata[mesh_name] = mesh_data
    self.set_vertices(mesh_name, mesh_data.points)
    
    # set the color to be the meshes' initial location, and never change the color
    colors = vis.utils.color_mesh(mesh_data.points.T)
    
    # Color the vertex
    mesh_data.point_data.set_scalars(colors)

    mesh = self.pv_plotter.add_mesh(mesh_data, rgb=True, opacity = self.surface_opacity, name=mesh_name)
    
    mesh.user_matrix = self.transformation_matrix
    
    actor, _ = self.pv_plotter.add_actor(mesh, pickable=True, name=mesh_name)
    
    # Save actor for later
    self.mesh_actors[mesh_name] = actor
""" 

# meshobj.vertices = vertices # mesh.vertices.T / meshobj.sz.reshape((-1, 1))

def center_meshes(self, paths: Dict[str, (pathlib.Path or pv.PolyData)]):
    assert self.reference is not None, "Need to set the self.reference name first!"

    other_meshes = {}
    for id, obj in self.mesh_polydata.items():
        center = np.mean(obj.points, axis=0)
        obj.points -= center
        if id == self.reference:
            reference_center = center.copy()
            # vis.utils.writemesh(paths[id], obj.points.T, center=True)
        else:
            other_meshes[id] = center

    # add the offset
    for id, center in other_meshes.items():
        offset = center - reference_center
        self.mesh_polydata[id].points += offset
        # vis.utils.writemesh(paths[id], self.mesh_polydata[id].points.T, center=True)

    print('hhhh')

app.center_meshes({'ossicles': ossicles_path, 'facial_nerve': facial_nerve_path, 'chorda': chorda_path})

# if 'centered' in name: name = '_'.join(name.split("_")[:-1]) + suffix
# else: name += suffix

def center_meshes(self, paths: Dict[str, (pathlib.Path or pv.PolyData)]):
    assert self.reference is not None, "Need to set the self.reference name first!"

    other_meshes = {}
    for id, obj in self.mesh_polydata.items():
        center = np.mean(obj.points, axis=0)
        obj.points -= center
        if id == self.reference:
            reference_center = center.copy()
            # vis.utils.writemesh(paths[id], obj.points.T, center=True)
        else:
            other_meshes[id] = center

    # add the offset
    for id, center in other_meshes.items():
        offset = center - reference_center
        self.mesh_polydata[id].points += offset
        # vis.utils.writemesh(paths[id], self.mesh_polydata[id].points.T, center=True)

    print('hhhh')

def test_flip_left_ossicles_color(app):

    ossicles_path = vis.config.OSSICLES_MESH_PATH_6742_left

    app.set_transformation_matrix(np.eye(4))

    app.load_meshes({'ossicles': ossicles_path})
    
    app.plot()

    rendered_mask_path = vis.config.DATA_DIR / "rendered_mask" / "rendered_mask_whole_6742.png"
    rendered_mask = np.array(PIL.Image.open(rendered_mask_path)) / 255

    modified_mask = np.where(rendered_mask != [0, 0, 0], 1 - rendered_mask, rendered_mask)
    modified_mask = modified_mask[:, ::-1, ...]

    vertices = getattr(app, f'ossicles_vertices')
    pts2d, pts3d = vis.utils.create_2d_3d_pairs(modified_mask, vertices)
    predicted_pose = vis.utils.solve_epnp_cv2(pts2d, pts3d, app.camera_intrinsics, app.camera.position)

    print("hhhh")

# ~ config.py
# gt_pose_455_right = np.array([[  0.36189961,   0.31967712,  -0.87569128,   5.33823202],
#                             [  0.40967285,   0.78925644,   0.45743024, -32.5816239 ],
#                             [  0.83737496,  -0.52429077,   0.15466856,  12.5083066 ],
#                             [  0.,           0.,           0.,           1.        ]])

# gt_pose_5997_right = np.array([[  -0.14498174,   -0.34676691,   -0.92667849,   29.36436624],
#                         [   0.33413722,    0.86439266,   -0.3757361,   -13.54538251],
#                         [   0.93130693,   -0.36411267,   -0.00945343, -104.0636636 ],
#                         [   0.,            0.,            0.,            1.        ]])


# gt_pose_6088_right = np.array([[  0.36049218,  -0.12347807,  -0.93605796,  17.37936422],
#                         [  0.31229879,   0.96116227,  -0.00651795, -27.17513405],
#                         [  0.89102231,  -0.28692541,   0.38099733, -19.1631882 ],
#                         [  0.,           0.,           0.,           1.        ]])


# gt_pose_6108_right = np.array([[  0.20755796,   0.33304378,  -0.9197834,   10.89388084],
#                         [  0.61199071,   0.68931778,   0.38769624, -36.58529423],
#                         [  0.76314289,  -0.64336834,  -0.06074633, 229.45832825],
#                         [  0.,           0.,           0.,           1.        ]]) #  GT pose

# gt_pose_6742_left = np.array([[ -0.00205008,  -0.27174699,   0.96236655, -18.75660285],
#                         [ -0.4431008,    0.86298269,   0.24273971, -13.34068231],
#                         [ -0.89646944,  -0.42592774,  -0.1221805,  458.83536963],
#                         [  0.,           0.,           0.,           1.        ]]) #  GT pose

# gt_pose_6742_right = np.eye(4)

# gt_pose_632_right = np.array([[  0.01213903,  -0.23470041,  -0.97199196,  23.83199935],
#                             [  0.7709136,    0.62127575,  -0.14038752, -19.05412711],
#                             [  0.63682404,  -0.74761766,   0.18847542, 602.2021275 ],
#                             [  0.,           0.,           0.,           1.        ]])

# gt_pose_6087_left = np.array([[  0.20370912,  -0.21678892,   0.95472779, -20.79224732],
#                                 [ -0.62361071,   0.72302221,   0.29723487,  -4.9027381 ],
#                                 [ -0.75472663,  -0.65592793,   0.01209432,  30.25550556],
#                                 [  0.,           0.,           0.,           1.        ]])

# gt_pose_6320_right = np.array([[  0.13992712,  -0.06843788,  -0.98779384,  19.45358842],
#                             [  0.50910393,   0.86061441,   0.01249129, -27.0485824 ],
#                             [  0.84925473,  -0.50463759,   0.15526529, 305.97544605],
#                             [  0.,           0.,           0.,           1.        ]])

# gt_pose_6329_right = np.array([[  0.09418739,   0.38382761,  -0.91858865,   7.3397235 ],
#                                 [  0.66883124,   0.65905617,   0.34396181, -33.31646256],
#                                 [  0.73742355,  -0.64677765,  -0.19464113, 302.59170409],
#                                 [  0.,           0.,           0.,           1.        ]])

# gt_pose_6602_right = np.array([[  0.22534241,   0.44679594,  -0.86579107,   3.33442317],
#                                 [  0.49393868,   0.7135863,    0.496809,   -28.9554841 ],
#                                 [  0.83978889,  -0.53959983,  -0.05988854, 299.38210116],
#                                 [  0.,           0.,           0.,           1.        ]])

# gt_pose_6751_right = np.array([[  0.14325502,  -0.47155627,  -0.87012222,  22.98132783],
#                                 [  0.88314596,   0.45772661,  -0.10266232, -23.0138221 ],
#                                 [  0.44668916,  -0.75373804,   0.48202465, 137.22705977],
#                                 [  0.,           0.,           0.,           1.        ]])

# if self.mirror_objects:
    #     mesh_name = mesh_name + '_mirror'

    #     # create a DEEP copy
    #     mesh_data = mesh_data.copy(deep=True)
    #     # mesh_data.points = (np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ mesh_data.points.T).T

    #     # mirror the object based on the origin (0, 0, 0)
    #     # mesh_data = mesh_data.reflect((1, 0, 0), point=(0, 0, 0)) 
        
    #     # Save the mesh data to dictionary
    #     self.mesh_polydata[mesh_name] = mesh_data
    #     # set vertices attribute
    #     self.set_vertices(mesh_name, mesh_data.points)
    #     # Color the vertex: set the color to be the meshes' initial location, and never change the color
    #     # colors = vis.utils.color_mesh(mesh_data.points.T)
    #     colors = vis.utils.color_mesh(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ mesh_data.points.T)
    #     mesh_data.point_data.set_scalars(colors)
    #     mesh = self.pv_plotter.add_mesh(mesh_data, rgb=True, opacity=self.surface_opacity, name=mesh_name)
    #     # Set the transformation matrix to be the mesh's user_matrix
    #     self.transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ self.transformation_matrix
    #     mesh.user_matrix = self.transformation_matrix
    #     self.initial_poses[mesh_name] = self.transformation_matrix

    #     # Add and save the actor
    #     actor, _ = self.pv_plotter.add_actor(mesh, pickable=True, name=mesh_name)
    #     self.mesh_actors[mesh_name] = actor

color_mask_binarized = vis.utils.color2binary_mask(color_mask_whole)
binary_mask = color_mask_binarized * seg_mask
color_mask = (color_mask_whole * seg_mask).astype(np.uint8)
assert (binary_mask == vis.utils.color2binary_mask(color_mask)).all(), "render_binary_mask is not the same as converted render_color_mask"

downscale_binary_mask = cv2.resize(binary_mask, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

 # # torch implementation
# trans = torchvision.transforms.Resize((h, w))
# color_mask = trans(torch.tensor(downscale_color_mask).permute(2,0,1))
# color_mask = color_mask.permute(1,2,0).detach().cpu().numpy()
# binary_mask = trans(torch.tensor(downscale_binary_mask).unsqueeze(-1).permute(2,0,1))
# binary_mask = binary_mask.permute(1,2,0).squeeze().detach().cpu().numpy()

# make sure the binary mask only contains 0 and 1
binary_mask = np.where(binary_mask != 0, 1, 0)
binary_mask_bool = binary_mask.astype('bool')
assert (binary_mask == binary_mask_bool).all(), "binary mask should be the same as binary mask bool"

plt.subplot(223)
plt.imshow(binary_mask)

def color2binary_mask(color_mask):
    binary_mask = np.zeros(color_mask[...,:1].shape)
    x, y, _ = np.where(color_mask != [0., 0., 0.])
    binary_mask[x, y] = 1      
    return 

# binary_mask = color2binary_mask(color_mask)
        
# make sure the binary mask only contains 0 and 1
binary_mask = np.where(binary_mask != 0, 1, 0)
binary_mask_bool = binary_mask.astype('bool')
assert (binary_mask == binary_mask_bool).all(), "binary mask should be the same as binary mask bool"

# To convert color_mask to bool type, we need to consider all three channels for color image, or conbine all channels to grey for color images!
color_mask_bool = (0.3*color_mask[..., :1] + 0.59*color_mask[..., 1:2] + 0.11*color_mask[..., 2:]).astype("bool") 
# # solution2
# color_mask_bool = np.logical_or(color_mask.astype("bool")[..., :1], color_mask.astype("bool")[..., 1:2], color_mask.astype("bool")[..., 2:])
# # solution3
# color_mask_bool = color_mask.astype("bool")
# color_mask_bool = (color_mask_bool[..., :1] + color_mask_bool[..., 1:2] + color_mask_bool[..., 2:]).astype("bool")
assert (binary_mask == color_mask_bool).all(), "binary_mask is not the same as the color_mask_bool"

if npts == -1:
    rand_pts = pts
else:
    rand_pts_idx = np.random.choice(pts.shape[0], npts)
    rand_pts = pts[rand_pts_idx,:]

# self.pv_plotter.off_screen = self.off_screen
# if not self.pv_plotter.off_screen:

# mesh_data.point_data.set_scalars(colors)
# mesh = self.pv_plotter.add_mesh(mesh_data, rgb=True, opacity=self.surface_opacity, name=mesh_name) #, show_edges=True)
            
# save the rendered whole image
# vis.utils.save_image(color_mask_whole, vis.config.OUTPUT_DIR / "rendered_mask", f"rendered_mask_whole_{name}.png")
# save the rendered partial image
# if hand_draw_mask is not None: vis.utils.save_image(color_mask, vis.config.OUTPUT_DIR / "rendered_mask", f"rendered_mask_partial_{name}.png")

    def render_scene(self, render_image:bool, image_source:np.ndarray=None, scale_factor:Tuple[float] = (0.01, 0.01, 1), render_objects:List=[], surface_opacity:float=1, return_depth_map: bool=False):
        
        pv_render = pv.Plotter(window_size=[self.window_size[0], self.window_size[1]], lighting=None, off_screen=True)
        pv_render.enable_joystick_actor_style()
 
        if render_image:
            assert image_source is not None, "image source cannot be None!"
            image = pv.UniformGrid(dimensions=(1920, 1080, 1), spacing=scale_factor, origin=(0.0, 0.0, 0.0))
            image.point_data["values"] = image_source.reshape((1920*1080, 3)) # order = 'C
            image = image.translate(-1 * np.array(image.center), inplace=False)
            pv_render.add_mesh(image, rgb=True, opacity=1, name="image")
        else:
            # background set to black
            pv_render.set_background('black')
            assert pv_render.background_color == "black", "pv_render's background need to be black"
            
            # Render the targeting objects
            for object in render_objects:
                mesh_data = self.mesh_polydata[object]
                colors = vis.utils.color_mesh(mesh_data.points) if not self.mirror_objects else vis.utils.color_mesh(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ mesh_data.points)
                mesh = pv_render.add_mesh(mesh_data,
                                        scalars=colors, 
                                        rgb=True, 
                                        style='surface',
                                        opacity=surface_opacity)
                mesh.user_matrix = self.transformation_matrix if not self.mirror_objects else np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ self.transformation_matrix
        
        pv_render.camera = self.camera.copy()
        pv_render.disable()
        pv_render.show()

        # obtain the rendered image
        rendered_image = pv_render.last_image
        # obtain the depth map
        depth_map = pv_render.get_image_depth()
              
        return rendered_image if not return_depth_map else (rendered_image, depth_map)
    
    def render_scene_point_clouds(self, render_image:bool, image_source:np.ndarray=None, scale_factor:Tuple[float] = (0.01, 0.01, 1), render_objects:List=[], surface_opacity:float=1, return_depth_map: bool=False):
        
        pv_render = pv.Plotter(window_size=[self.window_size[0], self.window_size[1]], lighting=None, off_screen=True)
        pv_render.enable_joystick_actor_style()
 
        if render_image:
            assert image_source is not None, "image source cannot be None!"
            image = pv.UniformGrid(dimensions=(1920, 1080, 1), spacing=scale_factor, origin=(0.0, 0.0, 0.0))
            image.point_data["values"] = image_source.reshape((1920*1080, 3)) # order = 'C
            image = image.translate(-1 * np.array(image.center), inplace=False)
            pv_render.add_mesh(image, rgb=True, opacity=1, name="image")
        else:
            # background set to black
            pv_render.set_background('black')
            assert pv_render.background_color == "black", "pv_render's background need to be black"
            
            # Render the targeting objects
            for object in render_objects:
                mesh_data = self.mesh_polydata[object]
                colors = vis.utils.color_mesh(mesh_data.points) if not self.mirror_objects else vis.utils.color_mesh(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ mesh_data.points)
                mesh = pv_render.add_mesh(mesh_data,
                                        scalars=colors,
                                        rgb=True, 
                                        style='points', 
                                        point_size=1, 
                                        render_points_as_spheres=False,
                                        opacity=surface_opacity)
                mesh.user_matrix = self.transformation_matrix if not self.mirror_objects else np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ self.transformation_matrix
        
        pv_render.camera = self.camera.copy()
        # pv_render.disable()
        pv_render.show()
        depth_map = pv_render.get_image_depth()

        # obtain the rendered image
        rendered_image = pv_render.last_image
        # obtain the depth map
          
        return rendered_image if not return_depth_map else (rendered_image, depth_map)

# color_mask_whole = app.render_scene(render_image=False, render_objects=['ossicles'])
    
# get the atlas mesh
# atlas_mesh = vis.utils.load_trimesh(vis.config.ATLAS_OSSICLES_MESH_PATH)
# atlas_mesh.vertices = trimesh.sample.sample_surface(mesh_source, 10000000)[0]
# atlas_mesh = pv.wrap(atlas_mesh)

# dist_mat = distance_matrix(mesh_data.points, mesh_5997.points)
# min_ind = dist_mat.argmin(axis=1)
# colors = vis.utils.color_mesh(mesh_5997.points) if not self.mirror_objects else vis.utils.color_mesh(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ mesh_data.points)
# colors = colors[min_ind, :]

# Color the vertex: set the color to be the meshes' initial location, and never change the color
# colors = vis.utils.color_mesh(atlas_mesh.points) if not self.mirror_objects else vis.utils.color_mesh(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ mesh_data.points)
# colors = vis.utils.color_mesh(mesh_data.points) if not self.mirror_objects else vis.utils.color_mesh(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ mesh_data.points)
