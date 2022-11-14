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
