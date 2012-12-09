"""
contour_irregular_data.py

This module/function lets you plot irregularlly spaced data by using an interpoltation scheme

Code taken/hacked modified from http://www.scipy.org/Cookbook/Matplotlib/Gridding_irregularly_spaced_data
"""


import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def contour(x,y,z, lines = 18, linewidth = 2):
    """
    x,y,z must be 1d arrays.
    """
    
    assert x.shape[0] == y.shape[0] == z.shape[0], "arrays x,y,z must be the same size"
    
    #make a grid that surrounds x,y support
    xi = np.linspace(x.min(),x.max(),100)
    yi = np.linspace(y.min(),y.max(),100)
    # grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    # contour the gridded data, plotting dots at the randomly spaced data points.
    plt.figure()
    CS = plt.contour(xi,yi,zi,18,linewidths=2)
    plt.clabel(CS, inline=1, fontsize=10)

    # plot data points.
    plt.scatter(x,y,marker='o',c=z,s=20, alpha = 0.8)
    plt.xlim(x.min(),x.max())
    plt.ylim(y.min(),y.max())
    plt.show()