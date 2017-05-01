# Copyright (c) 2017, Gerti Tuzi
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Gerti Tuzi nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color, exposure
from sklearn.preprocessing import StandardScaler


def CLAHENormalize(img):
    """
    Adaptive contrast normalize image. Expects a BGR input image.
    Refer to: https://www.mathworks.com/help/images/ref/adapthisteq.html
    :param img:image to normalize 
    :return: 
    normalized image
    """
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # equalize the histogram of the Y channel
    clahe = cv2.createCLAHE(clipLimit=0.005, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    # convert the LAB image back to BGR format
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)


def convert_color(img, color_space='BGR'):

    """
    Convert image to target color_space. 
    The image is assumed to be originally in BGR format (as read by cv2.imread())
    :param img: original BGR image
    :param color_space: target color space to convert to 
    :return: 
    copy of image in the target channel space
    """

    if color_space != 'BGR':
        if color_space == 'RGB':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else:
        feature_image = np.copy(img)

    return feature_image

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Draw bounding boxes, and return a copy of the image with boxes drawn onit
    :param img: source image to use
    :param bboxes: list of bounding boxes [( (x1, y), (x2, y2) ), (..) ...]
    :param color: color
    :param thick: thickness of line of boxes
    :return: 
    copy of image with boxes drawn on it
    """
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def draw_labeled_bboxes(img, labels):
    """
    Use labels image and put bounding boxes around the labeled regions.
    'labels' are obtained from "scipy.ndimage.measurements import label" module, where
    labels = label(heatmap)
    :param img: image
    :param labels: labels
    :return: 
    Copy of image with labels drawn on it
    """

    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def find_matches(img, template_list, method=cv2.TM_SQDIFF):
    """
    # Other options include: cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCORR',
    #         'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
    :param img: image to match the template on
    :param template_list: template list
    :param method: method template matching 
    :return: 
    list of bounding boxes where box is defined as:
    ((x1, y1), (x2, y2))
    """
    # Iterate over the list of templates
    bboxes = [] # resulting values
    for tmp in template_list:
        # Apply template Matching
        # search the image for the template
        tmpimg =cv2.imread(tmp)
        _, w, h = tmpimg.shape[::-1]
        res = cv2.matchTemplate(img, tmpimg, method)
        # extract the location of the best match in each case
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if (method == cv2.TM_SQDIFF) or (method == cv2.TM_SQDIFF_NORMED):
            # Min desired for a match
            bboxes.append((min_loc, tuple(map(lambda x, y: x + y, min_loc, (w, h)))))
        elif (method == cv2.TM_CCORR) or (method == cv2.TM_CCORR_NORMED):
            # Max desired for a match
            bboxes.append((max_loc, tuple(map(lambda x, y: x + y, max_loc, (w, h)))))
        elif (method == cv2.TM_CCOEFF) or (method == cv2.TM_CCOEFF_NORMED):
            # Max desired for a match
            bboxes.append((max_loc, tuple(map(lambda x, y: x + y, max_loc, (w, h)))))
        else:
            raise Exception('Incorrect method used')

    # Return the list of bounding boxes
    return bboxes

def imghist(img, nobins = 32, bin_range=(0, 256)):
    """
    Generate histogram of image. 
    :param img: image to process
    :param nobins: number of bins in histogram
    :param bin_range: min/max values of bins
    :param hist_feats: each channel's histogram (counts) 
                       concetenated into one feature vector
    :return: 
    bin count, bin centers, hist_feats
    """

    # Iterate over each channel and generate a bin for each
    hist = []
    hist_feats = []
    bin_edges = []
    bin_centers = []
    for c in range(0, img.shape[2]):
        hist.append(np.histogram(img[:, :, c], bins=nobins, range=bin_range))
        # histograms as features (counts)
        hist_feats.append(hist[-1][0])
        # edges
        bin_edges.append(hist[-1][1])
        # centers
        bin_centers.append((bin_edges[-1][1:] + bin_edges[-1][0:len(bin_edges[-1])-1])/2)

    return hist, bin_edges, bin_centers, np.concatenate(hist_feats)

def bin_spatial(img, color_space='BGR', size=(32, 32)):
    """
    Spatial feature... resize and flatten
    Expects image returned from cv2.imread() --> 'BGR'
    
    :param img: Image being operated on. Expects image returned from cv2.imread() --> 'BGR'
    :param color_space: Color space to operate on
    :param size: resize image size (target size)
    :return: 
    """
    feature_image = convert_color(img, color_space)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features

def data_look(car_list, notcar_list):
    """
    # Function to return some characteristics of the dataset    
    :param car_list: 
    :param notcar_list: 
    :return: 
    """
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = cv2.imread(car_list[0]).shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = cv2.imread(car_list[0]).dtype
    # Return data_dict
    return data_dict

def single_img_features(img, color_space='BGR', spatial_size=(32, 32),
                        hist_bins=32, orient=9,pix_per_cell=8,
                        cell_per_block=2, hog_channel=0,
                        feats2gen=['bin_spat', 'hist', 'hog']):

    """
        Define a function to extract features from a single image window
        This function is very similar to extract_features()
        just for a single image rather than list of images    
    :param img: Image to process 
    :param color_space: clor space dimensions
    :param spatial_size: image target size. target size of image to generate spatial features for
                         Spatial features are pretty much pixels of a resized image and then flattened.
    :param hist_bins: number of bins for the histogram feature
    :param orient: number or orientations for HOG
    :param pix_per_cell: pixels per cell for HOG
    :param cell_per_block: cells per block for HOG
    :param hog_channel: number of channels to generate HOG over
    :param feats2gen: (string) list of features to generate. Options:
                      'bin_spat', 'hist', 'hog'. Default feats2gen = ['bin_spat', 'hist', 'hog']
    :return: 
    """

    #1) Define an empty list to receive features
    img_features = []

    #2) Apply color conversion
    feature_image = convert_color(img, color_space)

    # 3) Compute HOG features if flag is set
    if 'hog' in feats2gen:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        # 4) Append features to list
        img_features.append(hog_features)

    # 5) Compute spatial features if flag is set
    if 'bin_spat' in feats2gen:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #6) Append features to list
        img_features.append(spatial_features)

    # 7) Compute histogram features if flag is set
    if 'hist' in feats2gen:
        _, _, _,hist_features = imghist(feature_image, nobins=hist_bins)
        #8) Append features to list
        img_features.append(hist_features)


    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='BGR', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256), scale=False,
                     hog_channel=0, orient=9, pix_per_cell=8, cell_per_block=2,
                     feats2gen=['bin_spat', 'hist', 'hog'], norm = False):
    """
        Function to extract features from a list of images
    :param imgs: image file names
    :param color_space: color space ('BGR' default, as returned from cv2.imread())
    :param spatial_size: target size for anlysis window
    :param hist_bins: number of bins for histograms
    :param hist_range: range of values to scale histogram by
    :param scale: False(default), scale each feature before concatentation into the 
                  collective feature vector
    :param hog_channel: HOG channel to operate on. 'ALL' for all channels of the image
    :param orient: HOG feature orientations (bins)
    :param pix_per_cell: pixels for each cell to compute the HOG features
    :param cell_per_block: cells per block (HOG)
    :param feats2gen: (string) list of features to generate. Options:
                      'bin_spat', 'hist', 'hog'. Default feats2gen = ['bin_spat', 'hist', 'hog']
    :param norm: normalize (Fale default)
    :return:
     Feature vector, scaler (if scaling has been flagged, nothing (not None), otherwise)
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for imgf in imgs:
        # Read in each one by one
        img = cv2.imread(imgf)

        # Contrast normallize image
        if norm:
            img = CLAHENormalize(img)

        feature_image = convert_color(img, color_space)

        feats = []

        if 'hog' in feats2gen:
            if hog_channel == 'ALL':
                hog_ft = []
                for channel in range(feature_image.shape[2]):
                    hog_ft.append(get_hog_features(feature_image[:, :, channel],
                                                   orient, pix_per_cell, cell_per_block,
                                                   vis=False, feature_vec=True))
                hog_ft = np.ravel(hog_ft)
            else:
                hog_ft = get_hog_features(feature_image[:, :, hog_channel], orient,
                                          pix_per_cell, cell_per_block,
                                          vis=False, feature_vec=True)
            feats.append(hog_ft)

        # Apply bin_spatial() to get spatial color features
        if 'bin_spat' in feats2gen:
            bin_spat_ft = bin_spatial(feature_image, size=spatial_size)
            feats.append(bin_spat_ft)

        if 'hist' in feats2gen:
            # Apply color_hist() to get color histogram features
            _, _, _, hist_ft = imghist(feature_image, nobins=hist_bins, bin_range=hist_range)
            feats.append(hist_ft)

        # Concatenate features into one bigger vector
        feats = np.concatenate(feats)
        # Append the new feature vector to the features list
        features.append(feats)

    features = np.array(features)
    if scale:
        scaler = StandardScaler().fit(features)
        features = scaler.transform(features)
        return features, scaler
    else:
        return features


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Generate HOG features of image
    :param img: image to operate on
    :param orient: gradient orientations bins (binning of the angles obtained from gradient angles)
    :param pix_per_cell: pixel per defined cell, where HOG is generated
    :param cell_per_block: block is a set of cells
    :param vis: visualized image
    :param feature_vec: True (default) / False. Return the features as a vector
    :return:
     Features obtained, visualization image (or None if not requested)
    """
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), visualise=vis,
                                  feature_vector=feature_vec, transform_sqrt = True)
        return features, hog_image
    else:
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), visualise=vis,
                       feature_vector=feature_vec, transform_sqrt = True)

        return features


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
        Function that takes an image, start and stop positions in both x and y, 
        window size (x and y dimensions), and overlap fraction (for both x and y)
    :param img: image to operate on
    :param x_start_stop: x-dim start stop positions
    :param y_start_stop: y-dim start stop positions
    :param xy_window: window size (x, y) in pixels
    :param xy_overlap: overlap amount as a percentage of the sliding window
                       along respective dimensions. Values are 0 ~ 1
    :return: 
    List of windows generated through the slide
    Each window is a tuple as:
    ((startx, starty), (endx, endy))
    """

    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list



def search_windows(img, windows, clf, scaler, color_space='BGR',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, feats2gen = ['bin_spat', 'hist', 'hog']):
    """
    Define a function you will pass an image 
    and the list of windows to be searched (output of slide_windows())
    :param img: image to operate on
    :param windows: list of windows to be searched
    :param clf: classifier (must have .predict())
    :param scaler: scaler used for training the classifier
    :param color_space: color space of image. 
                        Convert to BGR if other than BGR
                        cv2 default space is BGR ... so there you have it.
    :param spatial_size: target size of image to generate spatial features
    :param hist_bins: number of bins for the histogram features
    :param hist_range: histogram range of the histogram features (0-256)
    :param orient: HOG number of orientations
    :param pix_per_cell: HOG pixels / cell
    :param cell_per_block: HOG cell / block
    :param hog_channel: HOG channel to operate on. 0, 1, 2, 'ALL'
    :param feats2gen: (string) list of features to generate. Options:
                      'bin_spat', 'hist', 'hog' 
    :return:
     List of windows of positive detections. Note that these are a subset of the windows
     generated from slide_window(). Where each window is ordered as:
     ((startx, starty), (endx, endy))
    """

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, feats2gen = feats2gen)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def find_objects(imgin, ystart, ystop, scale, svc, color_space,
                 X_scaler, orient, pix_per_cell, window, cells_per_step,
                 cell_per_block, spatial_size, hist_bins,
                 feats2gen = ['bin_spat', 'hist', 'hog'],
                 retlabels = False, norm = False):
    """
    Define a single function that can extract features using hog sub-sampling and make predictions.
    There is a slinding window operation. This window size must be the same size as the training patches
    presented to the classifier. The scaling factor, blows up / reduces the image size which effectively reduces/blows 
    up the window size (note the reversal of direction). The window move 'cells_per_step' cells
    :param img: image to process
    :param ystart: start search y-coordinates
    :param ystop: stop search y-coordinates
    :param scale: image scale/window scaling. 
                  Image is scaled as: image.shape/scale
                  Search window is scaled as: window_size * scale
    :param svc: SVM classifier
    :param color_space: color space to perform feature extraction in
    :param X_scaler: training data scaler
    :param orient: HOG orientations
    :param pix_per_cell: HOG pixels / cell
    :param window: (x, y) window size in pixels. This is the window size that was used to train the classifier
    :param cells_per_step: cell-steps by which the window moves
    :param cell_per_block: HOG cell / block
    :param spatial_size: spatial resizing for spatial features
    :param hist_bins: number of histogram bins for histogram features
    :param feats2gen: (string) list of features to generate. Options:
                      'bin_spat', 'hist', 'hog'. Default feats2gen = ['bin_spat', 'hist', 'hog']
    :param retlabels - return labels (i.e. bounding boxes) of images found. False (default)
    :param norm - normalize (False default)
    :return: 
        Image with the drawn detections. 
        Return labels if 'retlabels == True'
    """

    # Normalize image
    if norm:
        img = CLAHENormalize(np.copy(imgin))
    else:
        img = np.copy(imgin)

    draw_img = np.copy(img)
    # img = img.astype(np.float32) / 255

    # If none, search the whole image
    if ystart == None:
        ystart = 0
    if ystop == None:
        ystop = img.shape[0]

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, color_space=color_space)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    # nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    # cells_per_step = 2  # Instead of overlap, define how many cells to step
    #
    # nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    # nysteps = (nyblocks - nblocks_per_window) // cells_per_step


    # GT
    nblocks_per_window_x = (window[0] // pix_per_cell) - cell_per_block + 1
    nblocks_per_window_y = (window[1] // pix_per_cell) - cell_per_block + 1

    # GT: compute cells / window: where window is sized as (x_pixels, y_pixels)
    # So: x_pixels is pixel/window in the x-direction
    cell_per_window_x = window[0] // pix_per_cell #cell / window = (pix/wind) * 1/(pix/cell)
    cell_per_window_y = window[1] // pix_per_cell

    # GT: Number of cell-steps is the remainder from the window overlap
    cells_per_step_x = int(cells_per_step)
    cells_per_step_y = int(cells_per_step)

    nxsteps = (nxblocks - nblocks_per_window_x) // cells_per_step_x
    nysteps = (nyblocks - nblocks_per_window_y) // cells_per_step_y

    if 'hog' in feats2gen:
        # Pre-compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    if retlabels:
        labels = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step_y
            xpos = xb * cells_per_step_x
            feat_cum = ()
            if 'hog' in feats2gen:
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window_y, xpos:xpos + nblocks_per_window_x].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window_y, xpos:xpos + nblocks_per_window_x].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window_y, xpos:xpos + nblocks_per_window_x].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                feat_cum += (hog_features,)

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            # window is a tuple of window size in pixels (x, y)
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window[1], xleft:xleft + window[0]], window)

            # Get color features
            if 'bin_spat' in feats2gen:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                feat_cum += (spatial_features,)

            if 'hist' in feats2gen:
                _, _, _, hist_features = imghist(subimg, nobins=hist_bins)
                feat_cum += (hist_features,)

            # Scale features and make a prediction
            # Order of features matters here
            test_features = X_scaler.transform(np.hstack(feat_cum).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw_x = np.int(window[0] * scale)
                win_draw_y = np.int(window[1] * scale)

                bbox = ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw_x, ytop_draw + win_draw_y + ystart))
                cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 0, 255), 6)

                if retlabels:
                    labels.append(bbox)

    if retlabels:
        return draw_img, labels
    else:
        return draw_img


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    """
    Zero out pixels in heatmap below the threshold
    :param heatmap: 
    :param threshold: 
    :return:
     reference to the heatmap passed
    """
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def plot3d(pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)], ax = None):
    """
    Plot pixels in 3D in their channel spaces.
    """

    if ax is None:
        # Create figure and 3D axes
        fig = plt.figure(figsize=(8, 8))
        ax = Axes3D(fig)


    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation
