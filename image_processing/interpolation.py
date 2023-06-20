import numpy as np
import cv2

import util
import image_processing.filters as filter

##############################################################################################################
###                                         Transform
##############################################################################################################

#For Grayscale and RGB/HSV/Lab/etc.
def transformInterpolate(image, P, interpolator='bilinear'):
    #if P is homogeneous or not


    #compute grid points in new image (width, height, point)
    projGrid = image.shape
    newGrid = np.meshgrid((gridmin, gridmax))

    #find new image height and width, initialize
    height = 
    width = 
    new = np.zeros((width, height))

    #Define and init interpolation kernel
    new = None
    if interpolator == 'bilinear':
        
    elif interpolator == 'bicubic':
        pass
    elif interpolator == 'sinc':
        pass
    elif interpolator == 'lanczos':
        pass

    #apply the transformation using interpolation


    return new

##############################################################################################################
###                                         Interpolation Kernels
##############################################################################################################

#defines interpolation operator
#defines interpolation grid regime
#
class Interpolate:
    def __init__(self, kernel):
        pass

    #generate discrete grid kernel from function
    def uniform(self, shape=(3,3)):
        pass

    #
    def nonuniform(self):
        pass
    

##############################################################################################################
###                                         Interpolation Kernels
##############################################################################################################

#if uniform grid sample discrete kernel
#if nonuniform use piecewise function

#nearest neighbor kernel
def nearestNeighbor():
    pass

#bilinear interpolation kernel
def bilinear():
    pass

#bicubic interpolation kernel
def bicubic():
    pass

#sinc interpolation kernel
def sinc():
    pass

#def lanczos interpolation kernel
def lanczos():
    pass
