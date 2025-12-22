from image_loading.loader import ImageRaw, ImageProcessed
from image_processing.pre_processing import PreProcessor
from image_processing.edge_detection import EdgeDetector
from image_processing.segmentation import Segmentation
from shape_descriptors.shape_representation import ContourAnalysis
from utilities.histograms import Histogram
from contour_extraction.countour_extraction import Morphological, ContourExtraction, BoundingBoxCreation

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# STATIC PATHS

input_path = r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\inputs'
output_path = r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\outputs'
image = cv2.imread(os.path.join(input_path, "trial.jpg"))

filtered_image = image

cannee = EdgeDetector(filtered_image)
cannee.canny_scharr_gradient()
cannee.canny_non_max_suppression(wk_multiplier=0.1, str_multiplier=0.15)
edges = cannee.canny_hysteresis()
cv2.imwrite(os.path.join(output_path, "Canny_w_Scharr_2.jpg"), edges)

h = Histogram('Canny with Scharr - Non Max Suppression and Hysteresis', 'x', 'y')
counts = h.create_2d_patch_histogram(edges, patch_size=(50,50), count_mode='nonzero', threshold=None, output_path=output_path, show=True)