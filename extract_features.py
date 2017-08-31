import cv2
import numpy as np
import os
from skimage.feature import hog
from utils import read_image

def get_color_histogram(img, nbins=32, bins_range=(0, 256)):
    n_pix = img.shape[0] * img.shape[1]
    return np.concatenate([np.histogram(img[:,:,c], nbins, bins_range)[0] for c in range(img.shape[-1])])/n_pix


def extract_features_from_images(paths, cspace='RGB', spatial_size=None,
                     hist_bins=0,
                     hog_orient=9, hog_pix_per_cell = 8, hog_cell_per_block=2, hog_channel=0):
    features = []
    for path in paths:
        img = read_image(path, cspace)
        features.append(extract_features(img, spatial_size,
                                        hist_bins,
                                        hog_orient, hog_pix_per_cell,
                                        hog_cell_per_block, hog_channel))
    return np.array(features)


def extract_features(img, spatial_size=(32, 32),
                     hist_bins=0,
                     hog_orient=9, hog_pix_per_cell = 8, hog_cell_per_block=2, hog_channel=0):
    spatial_features = []
    if spatial_size is not None:
        # scale the image to spatial_size and vectorize it
        spatial_features = cv2.resize(img, spatial_size).ravel()
    color_features = []
    if hist_bins > 0:
        # compute the normalized color histogram
        color_features = get_color_histogram(img*255, hist_bins)
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(img.shape[2]):
            # use hog from skimage
            hog_features.append(hog(img[:,:,channel],
                                orientations=hog_orient,
                                pixels_per_cell=(hog_pix_per_cell, hog_pix_per_cell),
                                cells_per_block=(hog_cell_per_block, hog_cell_per_block),
                                transform_sqrt=True,
                                visualise=False,
                                feature_vector=True))

        hog_features = np.ravel(hog_features)
    elif hog_channel == 'NONE':
        hog_features = []
    else:
        hog_features = hog(img[:,:,hog_channel],
                           orientations=hog_orient,
                           pixels_per_cell=(hog_pix_per_cell, hog_pix_per_cell),
                           cells_per_block=(hog_cell_per_block,hog_cell_per_block),
                           transform_sqrt=True,
                           visualise=False,
                           feature_vector=True)
    features = np.concatenate((spatial_features, color_features, hog_features))
    return features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False,
                     feature_vec=True):
    '''
    Convenience function that wraps skimage's hog function
    '''
    return hog(img, orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
