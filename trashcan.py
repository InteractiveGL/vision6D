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