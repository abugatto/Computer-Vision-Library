import numpy as np
import cv2

import util
import image_processing.interpolation as interp
import transformations as tfs

##############################################################################################################
###                                        Homography Algorithms
##############################################################################################################

#projective transformation given the line at infinity
#assumes line is in homogeneous coordinates
def projective2Affine(image, infinity, interpolator='bilinear'):
    #compute the transformation with H_A being the inversion matrix
    Hinv = np.zeros((3,3))
    Hinv[0,0] = -1.0
    Hinv[1,1] = -1.0
    Hinv[2,:] = -infinity #assume [l0, l1, l2]

    #produce new image via interpolation
    affine = interp.transformInterpolate(image, Hinv.T, interpolator=interpolator)

    return affine, Hinv

#Find metric image from affine image given 2 sets of parallel lines
def affine2metric(image, orthogonalSet, interpolator='bilinear'):
    #construct dual conic matrix
    A = []
    b = []
    for l0, l1 in orthogonalSet:
        A.append(np.array([
            l0[0] * l1[0],
            (l0[0] * l1[1] + l0[1] * l1[0]) / 2,
            [l0[1] * l1[1]]
        ]))

        b.append(-[l0[1] * l1[1]])

    #compute dual conic to the circular points (invariant to transformations)
    S = np.linalg.lstsq(np.array(A), np.array(b))
    S = np.array([
        [S[0], S[1]],
        [S[1], 1],
    ])

    #Compute the Transformation using SVD: Cd = H*U*S*Ut*Ht
    U, Sig, Vt = np.linalg.svd(S)
    Ha = np.zeros((3,3))
    Ha[:2,:2] = U @ np.sqrt(Sig) @ Vt
    Ha[3,3] = 1.0

    #Produce new image
    metric = interp.transformInterpolate(image, Ha, interpolator=interpolator)

    return metric, Ha

#compute both above functions
def projective2metric(image, infinity, parallel0, parallel1, interpolator='bilinear'):
    affine, Hinv = projective2Affine(image, infinity, interpolator=interpolator)
    metric, Ha, _ = affine2metric(affine, parallel0, parallel1, interpolator=interpolator)

    return metric, Ha @ Hinv

#compute metric from 5 orthogonal line pairs
def fivePointHomography(image, orthogonalSet, interpolator='bilinear'):
    #construct dual conic matrix with lines: li = (l0, l1, l2)
    A = []
    for l0, l1 in orthogonalSet:
        A.append(np.array([
            l0[0] * l1[0],
            (l0[0] * l1[1] + l0[1] * l1[0]) / 2,
            l0[1] * l1[1],
            (l0[0] * l1[2] + l0[2] * l1[0]) / 2,
            (l0[1] * l1[2] + l0[2] * l1[1]) / 2,
            l0[2] * l1[2]
        ]))

    #compute dual conic to the circular points (invariant to transformations)
    Cd = np.linalg.lstsq(np.array(A), np.zeros((5,)))

    #construct dual conic matrix
    Cd = np.array([
        [Cd[0], Cd[1]/2, Cd[3]/2],
        [Cd[1]/2, Cd[2], Cd[4]/2],
        [Cd[3]/2, Cd[4]/2, Cd[5]]
    ])

    #Compute the Transformation using SVD: Cd = H*U*S*Ut*Ht
    U, S, _ = np.linalg.svd(Cd)
    S[3,3] = 0.0
    Hp = U @ S

    #Produce new image
    metric = interp.transformInterpolate(image, Hp, interpolator=interpolator)

    return metric, Hp

##############################################################################################################
###                                     Parameter Finding Algorithms
##############################################################################################################

#assume correspondences are [[2D, 3D]...]
def DLT(correspondences):
    n = correspondences.shape[0]

    #form 3D matrix
    points3D = correspondences.T[:,0]
    points3D.reshape((n, 4))

    #form 2D matrix and reshape each to skew symmetric
    points2D = correspondences.T[:,1]
    for i in range(n):
        #append (2,3) matrices -> remove scaliing term
        points2D.append(util.skew(points2D[i])[:1])
    points2D = np.array(points2D)

    #Contruct A with kronecker product
    A = []
    for i in range(n):
        A.append(np.kron(points3D[i], points2D[i]))
    A = np.array(A)
    
    #Run least squares and find eigenvector w/ smallest eigenvalue
    _, S, Vt = np.linalg.svd(A)
    eigmin = np.argmin(np.diag(S))
    P = Vt[:,eigmin]

    #Construct Projection matrix
    return P.reshape((3,4))

def eightPoint(correspondences):
    n = correspondences.shape[0]

    #Form point matrices
    points0 = correspondences.T[:,0]
    points1 = correspondences.T[:,1]

    #compute unit transforms
    com0 = np.sum(points0) / n
    scale0 = com0 / np.sqrt(2)
    T0 = np.array([
        [scale0, 0, -scale0 * com0[0]],
        [0, scale0, -scale0 * com0[1]],
        [0, 0, 1]
    ])

    com1 = np.sum(points1) / n
    scale1 = com1 / np.sqrt(2)
    T1 = np.array([
        [scale1, 0, -scale1 * com1[0]],
        [0, scale1, -scale1 * com1[1]],
        [0, 0, 1]
    ])

    #transform coordinates
    points0 = T0.get() @ points0
    points1 = T1.get() @ points1

    #Construct A with kronecker product
    A = np.kron(points0, points1)

    #Run least squares and enforce rank 2
    _, S, Vt = np.linalg.svd(A)
    eigmin = np.argmin(np.diag(S))
    F = Vt[:,eigmin]

    #Construct Fundamental matrix
    F.reshape((3,3))

    #approximate rank(F)=2
    U, S, Vt = np.linalg.svd(F)
    S[3,3] = 0.0
    F = U @ S @ Vt

    #Transform to normal scale: T1.T @ F @ T0
    return T1.T @ F @ T0


##############################################################################################################
###                                        Calibration
##############################################################################################################

#Implement 11 point algorithm for camera projection matrix
def calibrate():
    
    return P

#decompose P = KR into camera and orientation matrices
def decomposeProjection(self, P):

    return K, T

##############################################################################################################
###                                        Stereo Vision
##############################################################################################################

#decompose fundamental amtrix into 
def decomposeFundamental(F):


    return P0, X0, P1

#compute triangulation for point correspondences 
#using parallax method
def triangulate(F, correspondences):


    return points3D

#compute given world coordinate correspondences
def adjustOrientation(correspondences):


    return points3DWorld