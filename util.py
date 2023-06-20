import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as Lines
import cv2
import math

##############################################################################################################
###                                        OpenCV and Matplotlib
##############################################################################################################

#display points
def plotPoints(ax, points, color='red'):
    ax.scatter(points, color=color)

#display lines
def plotLines(ax, linelist, color='black', size=1000):
    if linelist is not None:
        for i in range(0, len(linelist)):
            #get rho and theta
            params = linelist[i,:]

            #compute unit vector of line
            dir = np.array([math.cos(params[1]), math.sin(params[1])])

            #compute line
            p = np.array([dir[0] * params[0], dir[1] * params[0]])
            pt1 = (int(p[0] - size*dir[1]), int(p[1] + size*dir[0]))
            pt2 = (int(p[0] + size*dir[1]), int(p[1] - size*dir[0]))

            #draw line
            ax.add_artist(Lines.Line2D([pt1[0],pt2[0]], [pt1[1],pt2[1]], color=color))

def plotPointCloud():
    pass

##############################################################################################################
###                                             Math
##############################################################################################################

#skew symmetric cross product matrix of vector
def skew(w):
    return np.array([
        [0, -w[2], w[1]], 
        [w[2], 0, -w[0]], 
        [-w[1], w[0], 0]
    ])

#normalize image
def map1(image):
    return image.astype(np.float32) / 255

#denormalize image
def map255(image):
    return (image * 255).astype(np.uint8)

#line intersections
def computIntersections(line):
    

    return np.array([])

#line intersections
def computeRadialIntersections(lines):#, xbound, ybound):
    intersections = []
    parallels = []
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            l0 = lines[i]
            l1 = lines[j]

            #define matrix system
            A = np.array([
                [np.cos(l0[1]), np.sin(l0[1])],
                [np.cos(l1[1]), np.sin(l1[1])]
            ])
            b = np.array([l0[0], l1[0]]).T

            #if singular then parallel
            try:
                intersection = np.linalg.solve(A, b)

                #xbound = intersection[0] > xbound and intersection[0] < ybound
                #ybound = intersection[1] > xbound and intersection[1] < ybound
                #if xbound and ybound:
                intersections.append(intersection)
            except:
                parallels.append(np.array([l0, l1]))

    return np.array(intersections), np.array(parallels)

