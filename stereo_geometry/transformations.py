import numpy as np
import cv2

import util
import interpolation as interp

##############################################################################################################
###                                        Pixelwise transform
##############################################################################################################

#assumes 2d image and nonhomogeneous transformation
def transform2d(image, tf):
    #reshape
    dims = image.shape
    im2d = np.reshape(image, (dims[0]*dims[1], dims[2]))

    #transform
    im2d = tf @ im2d.T

    #reshape
    im = np.reshape(im2d.T, dims)

    return im

##############################################################################################################
###                                        Homogeneous Coordinates
##############################################################################################################

class HomogeneousTransform:
    def __init__(self, tf=None, A=None, t=None, v=None, w=None):
        if tf:
            self.dims = tf.shape[0]

            #set submatrices
            self.setGeneral(tf)
        elif A or t:
            w = 0 if not w else w
            if self.v:
                self.setProjective(A, t, v, w)
            else:
                A = np.identity((t.shape[0], t.shape[0])) if not A else A
                t = np.zeros((A.shape[0],)) if not t else t
                self.setAffine(A, t, w)
        else:
            self.A = None
            self.t = None
            self.v = None

    def get(self):
        tf = None
        if self.A:
            tf = np.zeros((self.dims,self.dims))
            tf[:self.dims-1,:self.dims-1] = self.A
            tf[:self.dims-1,self.dims] = self.t
            tf[self.dims,:self.dims-1] = self.v
            tf[self.dims,self.dims] = self.w

        return tf

    def setEuclidean(self, R, t):
        self.dims = R.shape[0] + 1

        #Init input matrix
        if not self.A:
            self.A = np.zeros((self.dims,self.dims))
            self.t = np.zeros((self.dims-1,))
            self.v = np.zeros((1,self.dims-1))

        #set input matrix
        self.A = R
        self.t = t
        self.w= 1.0

    def setSimilarity(self, s, R, t, w):
        self.dims = R.shape[0] + 1

        #Init input matrix
        if not self.A:
            self.v = np.zeros((1,self.dims-1))

        #set input matrix
        self.A = s*R
        self.t = s*t
        self.w = w

    def setAffine(self, A, t, w):
        self.dims = A.shape[0] + 1

        #Init input matrix
        if not self.A:
            self.v = np.zeros((1,self.dims-1))

        #set input matrix
        self.A = A
        self.t = t
        self.w = w

    def setProjective(self, A, t, v, w):
        self.dims = A.shape[0] + 1

        #set input matrix
        self.A = A
        self.t = t
        self.v = v
        self.w = w

    def setGeneral(self, tf):
        self.dims = tf.shape[0]
        if self.dims == 2 or self.dims == 3:
            self.setProjective(
                tf[:self.dims-1,:self.dims-1],
                tf[:self.dims-1,self.dims],
                tf[self.dims,:self.dims-1],
                tf[self.dims,self.dims]
            )

    def __eq__(self, tf):
        return self == tf
    
    def T(self):
        A = self.A.T
        v = self.t.T
        t = self.v.T

        return HomogeneousTransform(A=A, t=t, v=v, w=self.w)

    def inv(self):
        return HomogeneousTransform(tf = np.linalg.inv(self.get()))
    
    #implements AT 
    def rightMultiply(self, tf):
        return self.get() @ tf
    
    #Implements TA
    def leftMultiply(self, tf):
        return tf @ self.get()