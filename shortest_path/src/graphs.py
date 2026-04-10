"""
# MIT License                                                                       #
#                                                                                   #
# Copyright (c) 2022 Dr. Daniel Alejandro Matoz Fernandez                           #
# University of Warsaw, Institute of Theoretical Physics                            #
#               fdamatoz@gmail.com                                                  #
# Permission is hereby granted, free of charge, to any person obtaining a copy      #
# of this software and associated documentation files (the "Software"), to deal     #
# in the Software without restriction, including without limitation the rights      #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         #
# copies of the Software, and to permit persons to whom the Software is             #
# furnished to do so, subject to the following conditions:                          #
#                                                                                   #
# The above copyright notice and this permission notice shall be included in all    #
# copies or substantial portions of the Software.                                   #
#                                                                                   #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     #
# SOFTWARE.                                                                         #
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import random

# npoints = 25
def generate_points(npoints, seed: int = 11):
    np.random.seed(seed) #to reproduce meshes remove for production

    points = np.random.randint(0,15, size=(npoints, 2))
    points = np.unique(points, axis=0)

    """
    Dirty way to ensure that there are npoints unique
    points in the mesh
    """
    max_iter = 100
    itera = 0
    while points.shape[0]-npoints<0:
        next_points = np.random.randint(0,10, size=(npoints-points.shape[0], 2))
        points = np.append(points, next_points, axis=0)
        points = np.unique(points, axis=0)
        # print(next_points.shape[0], points.shape[0])
        itera += 1
        if itera>max_iter:
            break
    npoints = points.shape[0]

    D = Delaunay(points)
    simplices = D.simplices
    return points, simplices

points, simplices = generate_points(25)
def show_plot(points, simplices):
    plt.figure()
    plt.triplot(points[:, 0], points[:, 1], simplices)
    plt.scatter(points[:, 0], points[:, 1], color='r')
    plt.savefig("graph.png")

