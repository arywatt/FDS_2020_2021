import numpy as np
from numpy import histogram as hist
import math



#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0,filteringpath)

import gauss_module



#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

  
    # build bins array
    bin_size = 255.0 / num_bins
    bins = [0]
    for _ in range(num_bins):
      next_bin = bins[-1] + bin_size
      bins.append(next_bin)

    bins = np.array(bins)
    #bins = bins / 255


    #build histogram
    hists = np.zeros(num_bins)
    for value in np.array(img_gray).flatten():
        b = int(value * (1.0 / 255) * num_bins)
        hists[b] += 1

    hists = hists/np.sum(hists)
    return hists, bins


#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'


    #Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))


    # Loop for each pixel i in the image 
    for i in range(img_color_double.shape[0]):
        for j in range(img_color_double.shape[1]):
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i

          r,g,b = [int(x* (1.0 / 255) * num_bins)  for x in img_color_double[i][j]]
          hists[r,g,b] += 1
          

          
       

    #Normalize the histogram such that its integral (sum) is equal 1
    #... (your code here)

    hists = hists / np.sum(hists)
   

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists



#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'


    #... (your code here)


    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    
    
       # Loop for each pixel i in the image 
    for i in range(img_color_double.shape[0]):
        for j in range(img_color_double.shape[1]):
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i

          r,g,b = [int(x* (1.0 / 255) * num_bins)  for x in img_color_double[i][j]]
          hists[r,g] += 1


    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    return hists


# to cap value
def capvalue(value):
  if value <= -6:
    return -6

  if value >= 6:
    return 6

  return value

#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    sigma = 3
    imgDx,imgDy = gauss_module.gaussderiv(img_gray,sigma)


    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))


    for xValue,yValue in zip(np.array(imgDx).flatten(),np.array(imgDy).flatten()):
        
        x = capvalue(xValue)
        y = capvalue(yValue)
        
        x,y = int(num_bins * (x/12.0)),int(num_bins * (y/12.0))
        
        hists[x,y] += 1

    hists = hists / np.sum(hists)


    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists



def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
  if hist_name == 'grayvalue':
    return normalized_hist(img, num_bins_gray)
  elif hist_name == 'rgb':
    return rgb_hist(img, num_bins_gray)
  elif hist_name == 'rg':
    return rg_hist(img, num_bins_gray)
  elif hist_name == 'dxdy':
    return dxdy_hist(img, num_bins_gray)
  else:
    assert False, 'unknown distance: %s'%hist_name

