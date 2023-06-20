import numpy as np
import cv2
import util

import stereo_geometry.transformations as tfs

##############################################################################################################
###                                             RGB <-> sRGB
##############################################################################################################

def bgr2rgb(image):
    #reverse order
    return image[:,:,::-1]

def rgb2srgb(image):
    imsRGB = util.map1(image).copy()

    #implement conversion
    mask = imsRGB <= 0.04045
    imsRGB[mask] = imsRGB[mask] / 12.92
    imsRGB[~mask] = np.float_power((imsRGB[~mask] + 0.055), 2.4) / 1.055

    return util.map255(imsRGB)

def srgb2rgb(image):
    imRGB = util.map1(image).copy()

    #implement conversion
    mask = imRGB <= 0.0031308
    imRGB[mask] = imRGB[mask] * 12.92
    imRGB[~mask] = (np.float_power(imRGB[~mask], 1/2.4) * 1.055) - 0.055

    return util.map255(imRGB)

##############################################################################################################
###                                             sRGB <-> HSV
###                                             RGB <-> HSV
##############################################################################################################

#map XYZ to [0,1]
def normHSV(image):
    #define max values for each channel
    max = np.array([0.95047, 1.0, 1.08883])

    #nromalize
    image = image.astype(np.float32)
    for i in range(3):
        image[:,:,i] /= max[i]

    return image

#expects [0,255]
def rgb2hsv(image):
    imRGB = util.map1(image).copy()
    im = imRGB

    
    return util.map255(normHSV(im))

#expects [0,255]
def hsv2rgb(image):
    imHSV = normHSV(image).copy()
    im = imHSV
    
    return util.map255(util.map1(im))

def srgb2hsv(im):
    return rgb2hsv(srgb2rgb(im))

def hsv2srgb(im):
    return hsv2rgb(rgb2srgb(im))

##############################################################################################################
###                                             sRGB <-> XYZ
###                                             RGB <-> XYZ
##############################################################################################################

#map XYZ to [0,1]
def normXYZ(image):
    #define max values for each channel
    max = np.array([0.95047, 1.0, 1.08883])

    #nromalize
    image = image.astype(np.float32)
    for i in range(3):
        image[:,:,i] /= max[i]

    return image

#expects [0,255]
def rgb2xyz(image):
    imRGB = util.map1(image).copy()

    #transform matrix from RGB to XYZ
    T = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])

    #transform
    im = tfs.transform2d(imRGB, T)
    
    return util.map255(normXYZ(im))

#expects [0,255]
def xyz2rgb(image):
    imXYZ = normXYZ(image).copy()

    #transform matrix from RGB to XYZ
    T = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])

    #transform
    im = tfs.transform2d(imXYZ, T)
    
    return util.map255(util.map1(im))

def srgb2xyz(im):
    return rgb2xyz(srgb2rgb(im))

def xyz2srgb(im):
    return xyz2rgb(rgb2srgb(im))

##############################################################################################################
###                                             XYZ <-> Lab
###                                             sRGB <-> Lab
###                                             RGB <-> Lab
##############################################################################################################

WHITE = np.array([0.95047, 1.00000, 1.08883]) #D65 reference white

def normLab(image):
    image = image.astype(np.float32)
    image[:,:,0] /= 100.0
    image[:,:,1] += 128
    image[:,:,1] /= 255

    return image

#expects 255
def xyz2lab(image):
    #normalize
    imXYZ = util.map1(image).copy()

    #define CIE epsilon and kappa
    eps = 0.008856
    kappa = 903.3

    #define x,y,z channels w.r.t reference white
    x = imXYZ[:,:,0] / WHITE[0]
    y = imXYZ[:,:,1] / WHITE[1]
    z = imXYZ[:,:,2] / WHITE[2]

    #define f
    def func(x):
        f = np.zeros(x.shape)
        mask = x > eps
        f[mask] = np.float_power(x[mask], 1/3)
        f[~mask] = (kappa * x[~mask] + 16) / 116

        return f

    #define fx, fy, fz
    fy = func(y)

    #compute L, a, b
    imLab = np.zeros(image.shape)
    imLab[:,:,0] = 116 * fy - 16
    imLab[:,:,1] = 500 * (func(x) - fy)
    imLab[:,:,2] = 200 * (fy - func(z))

    return util.map255(normLab(imLab))

#expects 255
def lab2xyz(image):
    #normalize
    imLab = util.map1(image).copy()

    #define CIE epsilon and kappa
    eps = 0.008856
    kappa = 903.3

    #define lab image
    L = imLab[:,:,0]
    a = imLab[:,:,1]
    b = imLab[:,:,2]

    #compute L, a, b
    fy = (L + 16) / 116
    fx = (a / 500) + fy
    fz = fy - (b / 200)

    #define fy
    y = np.zeros(fy.shape)
    ymask = L > eps * kappa
    y[ymask] = fy[ymask]**3
    y[~ymask] = L[~ymask] / kappa

    #define fxz
    def func(f):
        x = np.zeros(f.shape)
        mask = f**3 > eps
        x[mask] = f[mask]**3
        x[~mask] = (116 * f[~mask] - 16) / kappa

        return f

    #define x,y,z channels w.r.t reference white
    imXYZ = np.zeros(image.shape)
    imXYZ[:,:,0] = func(fx) * WHITE[0]
    imXYZ[:,:,1] = y * WHITE[1]
    imXYZ[:,:,2] = func(fz) * WHITE[2]

    return util.map255(normXYZ(imXYZ))

def rgb2lab(im):
    return xyz2lab(rgb2xyz(im))

def lab2rgb(im):
    return xyz2rgb(lab2xyz(im))

def srgb2lab(im):
    return xyz2lab(srgb2xyz(im))

def lab2srgb(im):
    return xyz2srgb(lab2xyz(im))