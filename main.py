from image_loading.loader import ImageRaw, ImageProcessed
from image_processing.pre_processing import PreProcessor
from image_processing.edge_detection import EdgeDetector
from image_processing.segmentation import Segmentation
from shape_descriptors.shape_representation import ContourAnalysis
from utilities.histograms import Histogram
from contour_extraction.countour_extraction import Morphological, ContourExtraction

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# STATIC PATHS

input_path = r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\inputs'
output_path = r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\outputs'

# IMAGE LOADER

image = cv2.imread(os.path.join(input_path, "trial.jpg"))
exp_image = cv2.imread(os.path.join(output_path, "Filled_shapes.jpg"))

# IMAGE PRE-PROCESSING MODULE
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	

pre_processed_img = PreProcessor(image)
pre_processed_img.gaussian_blur()
pre_processed_img.bilateral_denoise()
prepim = pre_processed_img.histogram_equalization()

cv2.imwrite(os.path.join(output_path, "pre_processed.jpg"), prepim)	

# EDGE DETECTION

print("Start Edge Extraction")

preprosimg = cv2.imread(os.path.join(output_path, "pre_processed.jpg"))

def edge_detection_process():
	edges_det = EdgeDetector(preprosimg)
	cannyimg  = edges_det.canny_edge()
	cv2.imwrite(os.path.join(output_path, "canny.jpg"), cannyimg)
	
	sobelimg = edges_det.sobel()
	cv2.imwrite(os.path.join(output_path, "sobel.jpg"), sobelimg)
	
	scharrimgx = edges_det.scharr(dy=0)
	scharrimgy = edges_det.scharr(dx=0)
	scharrimg = (scharrimgx + scharrimgy)
	cv2.imwrite(os.path.join(output_path, "scharr.jpg"), scharrimg)
	
	prewittimg = edges_det.prewitt()
	cv2.imwrite(os.path.join(output_path, "prewitt.jpg"), prewittimg)
	return cannyimg, sobelimg, scharrimg, prewittimg

def area_segmentation(image_name):
	input_name = "seg_" + image_name
	edgeimg = cv2.imread(os.path.join(output_path, image_name))
	segmimg = Segmentation(edgeimg)
	seg = segmimg.threshold_otsu()
	cv2.imwrite(os.path.join(output_path, input_name), seg)
	# print(type(seg))
	# print(seg.shape)
	return seg
	
def segmenation_process():
	print("Area Segmentation happening")
	
	image_seg = area_segmentation("pre_processed.jpg")
	prewitt_seg = area_segmentation("prewitt.jpg")
	canny_seg = area_segmentation("canny.jpg")
	sobel_seg = area_segmentation("sobel.jpg")
	scharr_seg = area_segmentation("scharr.jpg")
	return image_seg, prewitt_seg, canny_seg, sobel_seg, scharr_seg
	
# MORPHOLOGICAL FILTERS

def morphing_time(bitimage, image_name):
	input_name = "morphed_" + image_name
	contor = ContourAnalysis(image, bitimage)
	morphed = contor.morph_preprocess()
	cv2.imwrite(os.path.join(output_path, input_name), morphed)

def morphing_process():	
	print("Morphological Filter being applied. Its morphing time.")
	morphing_time(cannyimg, "canny.jpg")
	morphing_time(sobelimg, "sobel.jpg")
	morphing_time(scharrimg, "scharr.jpg")
	morphing_time(np.invert(prewittimg), "prewitt.jpg")
	
	morphing_time(image_seg, "seg_image.jpg")
	morphing_time(np.invert(prewitt_seg), "seg_prewitt.jpg")
	morphing_time(canny_seg, "seg_canny.jpg")
	morphing_time(np.invert(sobel_seg), "seg_sobel.jpg")
	morphing_time(scharr_seg, "seg_scharr.jpg")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# cannyimg, sobelimg, scharrimg, prewittimg = edge_detection_process()
# image_seg, prewitt_seg, canny_seg, sobel_seg, scharr_seg = segmenation_process()
# morphing_process()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# HISTOGRAM GENERATION - GRAYSCALE & PREPROCESSED IMAGE
def histogram_process():
	img_list = [gray_image, prepim]
	labels_list = ['Grayscale Image','Preprocessed Image']
	hist_obj = Histogram("Grayscale - Pre-filtered")
	hist_obj.create_histogram(img_list, labels_list)

# SHAPE ANALSIS

edged_img = cv2.imread(os.path.join(output_path, "morphed_prewitt.jpg"))

contour_extraction_obj = ContourExtraction()
bin = contour_extraction_obj.preprocess(edged_img)
contours, binary = contour_extraction_obj.extract(bin)

print(f"Before eliminating smaller shapes: {len(contours)}")

value_list = []
for contour in contours:
	value_list.append(len(contour))

for index, contour in enumerate(contours):
	if len(contour) < 10:
		contours.pop(index)

print(f"After eliminating smaller shapes: {len(contours)}")

canvas = contour_extraction_obj.draw(image, contours)
cv2.imwrite(os.path.join(output_path, "Shape_analysis.jpg"), canvas)

canvas = contour_extraction_obj.fill(image, contours)
cv2.imwrite(os.path.join(output_path, "Filled_shapes.jpg"), canvas)

def morphological_processing(image):
	print("Erosion Process happening")
	morphing = Morphological(kernel_size=5)
	openings = morphing.open(image, iterations=1)
	closings = morphing.close(openings, iterations=1)
	cv2.imwrite(os.path.join(output_path, "Opening_example.jpg"), closings)

morphological_processing(exp_image)