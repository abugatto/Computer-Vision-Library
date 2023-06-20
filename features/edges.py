import numpy as np
import cv2

import util
import image_processing.filters as filter

##############################################################################################################
###                                        Linear Convolution Kernels
##############################################################################################################

#Prewitt edge detector
prewitt = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

#Sobel edge detector
sobel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

#laplacian of gaussian
def laplacianOfGaussian(sigma, dims):
    pass

##############################################################################################################
###                                             Canny Edge Detector
##############################################################################################################

#define canny class
class Canny:
    def __init__(self, ):
        pass

    def filter(self, im):
        pass