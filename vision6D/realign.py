"""
 RALIGN - Rigid alignment of two sets of points in k-dimensional
          Euclidean space.  Given two sets of points in
          correspondence, this function computes the scaling,
          rotation, and translation that define the transform TR
          that minimizes the sum of squared errors between TR(X)
          and its corresponding points in Y.  This routine takes
          O(n k^3)-time.

 Inputs:
   X - a k x n matrix whose columns are points 
   Y - a k x n matrix whose columns are points that correspond to
       the points in X
 Outputs: 
   c, R, t - the scaling, rotation matrix, and translation vector
             defining the linear map TR as 
 
                       TR(x) = c * R * x + t

             such that the average norm of TR(X(:, i) - Y(:, i))
             is minimized.
"""

"""
Copyright: Carlo Nicolini, 2013
Code adapted from the Mark Paskin Matlab version
from http://openslam.informatik.uni-freiburg.de/data/svn/tjtf/trunk/matlab/ralign.m 
"""

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