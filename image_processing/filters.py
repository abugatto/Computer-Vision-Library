import numpy as np
import cv2
import util

##############################################################################################################
###                                             Convolutions
##############################################################################################################

#define efficient convolution operation in numpy (mostly from stack overflow)
def conv2d(image, kernel):
    #Get dimensions of input and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    #Calculate padding dimensions
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    #Pad the image with zeros
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    #Create a "sliding window" view of the padded image using np.lib.stride_tricks.as_strided
    shape = (image_height, image_width, kernel_height, kernel_width)
    strides = padded_image.strides + padded_image.strides
    windows = np.lib.stride_tricks.as_strided(padded_image, shape=shape, strides=strides)

    #Perform convolution
    output = np.einsum('ijkl,kl->ij', windows, kernel)

    return output

##############################################################################################################
###                                        Linear Convolution Kernels
##############################################################################################################

#Average filter 
def average(n=3):
    return (1 / (n*n)) * np.ones((n, n))

##############################################################################################################
###                                        Gaussian Kernels
##############################################################################################################

#Gaussian Filter
def gaussian(sigma):
    pass

#define 2D gaussian:
def gaussian2d(sigma):
    pass

#define transformed gaussian
def gaussianTf(sigma, T):
    pass

##############################################################################################################
###                                        Nonlinear Convolution Kernels
##############################################################################################################

def median():
    pass

def bilateral():
    pass