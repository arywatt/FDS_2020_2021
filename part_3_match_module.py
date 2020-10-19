import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
    
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
    
    D = np.zeros((len(model_images), len(query_images)))
    
    
    # compute distance for each couple of query - image 
    for j ,query in enumerate(query_hists):
        for i,model in enumerate(model_hists):
            D[i,j] = dist_module.get_dist_by_name(model,query,dist_type)
        

    best_match = [] # to save best matches

    # for each query , find best model
    for j in range(len(query_images)):
        query_matches = D[:,j] # get query columns from matrix
        argmin = np.argmin(query_matches) # get index with minimum distance
        best_match.append(argmin) # save index for query

    best_match = np.array(best_match) # array of best match for each query

    
    return best_match,D



def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []

    # Compute hisgoram for each image and add it at the bottom of image_hist

    #... (your code here)
    for img in image_list :
        img_color = np.array(Image.open(img))
        img_color = img_color.astype('double')
        hist = histogram_module.get_hist_by_name(img=img_color,
                                                num_bins_gray=num_bins,
                                                hist_name=hist_type
                                                )
        image_hist.append(hist)

    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    
    
    plt.figure()

    num_nearest = 5  # show the top-5 neighbors
    
    #... (your code here)

    best_match , D = find_best_match(model_images=model_images,
                                    query_images=query_images,
                                    dist_type=dist_type,
                                    hist_type=hist_type,
                                    num_bins=num_bins
                                    )

    Q = len(query_images)
    pos = 0
    for j in range(Q):
        query_matches = D[:,j]
        best_args = np.argsort(query_matches)[:num_nearest]
        #best_match_imgs = [model_images[x] for x in best_args] 

        query_img = query_images[j]

        pos+=1
        plt.subplot(Q,6,pos); plt.imshow(np.array(Image.open(query_img)), vmin=0, vmax=255);plt.title(f'Q{j}')
        for ind in range(len(best_args)):
            pos+=1
            model_ind = best_args[ind]
            model_img = model_images[model_ind]
            plt.subplot(Q,6,pos); plt.imshow(np.array(Image.open(model_img)), vmin=0, vmax=255);plt.title(f'MO.{model_ind}')
            
    plt.show()

    

