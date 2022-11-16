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

def load_trimesh(self, filepath):
    with open(filepath, "rb") as fid:
        mesh = meshread(fid)
    orient = mesh.orient / np.array([1,2,3])
    mesh.vertices = mesh.vertices * np.expand_dims(mesh.sz, axis=1) * np.expand_dims(orient, axis=1)
    mesh = trimesh.Trimesh(vertices=mesh.vertices.T, faces=mesh.triangles.T)
    return mesh