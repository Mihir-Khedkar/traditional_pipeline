from image_loading.loader import ImageRaw, ImageProcessed
from image_processing.pre_processing import PreProcessor
from image_processing.edge_detection import EdgeDetector
from image_processing.segmentation import Segmentation
from shape_descriptors.shape_representation import ContourAnalysis
from utilities.histograms import Histogram
from contour_extraction.countour_extraction import Morphological, ContourExtraction, BoundingBoxCreation
from shape_analysis.feature_extractor import HoughLineDetector

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# STATIC PATHS

input_path = r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\inputs'
output_path = r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\outputs'
image = cv2.imread(os.path.join(input_path, "edges_trial_small.jpg"))

edges = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cannee = EdgeDetector(filtered_image)
# cannee.canny_scharr_gradient()
# cannee.canny_non_max_suppression(wk_multiplier=0.1, str_multiplier=0.15)
# edges = cannee.canny_hysteresis()
# cv2.imwrite(os.path.join(output_path, "Canny_w_Scharr.jpg"), edges)

# edges = cannee.canny_edge()
# cv2.imwrite(os.path.join(output_path, "Canny_w_sobel.jpg"), edges)

# h = Histogram('Canny with scharr - Non Max Suppression and Hysteresis', 'x', 'y')
# counts = h.create_2d_patch_histogram(edges, patch_size=(16,16), count_mode='nonzero', threshold=None, output_path=output_path, show=True)

print("doing a probabilistic Hough transform for feature extraction")

hough_object = HoughLineDetector(edges)
linesP = hough_object.detect_probabilistic(rho=1, theta=np.pi/180, threshold=500, min_line_length=250, max_line_gap=400)
hough_image = hough_object.draw_probabilistic(image, linesP, color=(0,255,0), thickness=2)
cv2.imwrite(os.path.join(output_path, "HoughLinesP_on_Canny.jpg"), hough_image)

print("doing a Hough transform for feature extraction")

lines = hough_object.detect_standard(rho=1, theta=np.pi/180, threshold=1000)
hough_image_std = hough_object.draw_standard(image, lines, color=(255,0,0), thickness=2)
cv2.imwrite(os.path.join(output_path, "HoughLines_on_Canny.jpg"), hough_image_std)