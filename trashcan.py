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