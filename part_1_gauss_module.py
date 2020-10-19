# code inside Gauss module
# Check if something is wrong 
# waiting for your corrections and suggestions

# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2



"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):
    x = [y for y in range(int(-3*sigma),int(3*sigma) +1)]
    x = np.array(x)
    Gx = np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.))) / np.sqrt(2*sigma*math.pi)
    return Gx, x



"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):

    # leverage gaussian
    gx, x = gauss(sigma)

    # normalize gaussian  vector
    gx *= 1.0 / gx.sum()

    # Reshape and compute convolution 
    gx = gx.reshape(1, gx.size)
    img_smooth_X = conv2(img, gx, 'same') # smooth on X axe 
    img_smoothed_X_Y = conv2(img_smooth_X, gx.T, 'same') # smooth on y axe

    return img_smoothed_X_Y



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):

    x = [y for y in range(int(-3 * sigma), int(3 * sigma) + 1)]
    x = np.array(x)
    Dx =  -np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.))) / np.sqrt(2 * (sigma**3) * math.pi)
    Dx = x*Dx
    return Dx,x



def gaussderiv(img, sigma):

    Gx, x = gauss(sigma)
    Dx, x = gaussdx(sigma)

    Gx = Gx.reshape(1, Gx.size)
    Dx = Dx.reshape(1, Dx.size)

    img_smoothed = gaussianfilter(img,sigma)

    #imgDx = conv2(conv2(img, Gx, 'same'), Dx, 'same')
    imgDx = conv2(img_smoothed, Dx, 'same')
    #imgDy = conv2(conv2(img, Gx.T, 'same'), Dx, 'same')
    imgDy = conv2(img_smoothed, Dx.T, 'same')

    return imgDx, imgDy


