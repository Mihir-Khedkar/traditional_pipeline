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

# IMAGE LOADER

image = cv2.imread(os.path.join(input_path, "trial.jpg"))
exp_image = cv2.imread(os.path.join(output_path, "seg_prewitt.jpg"))

# IMAGE PRE-PROCESSING MODULE

# EDGE DETECTION



def edge_detection_process():
	edges_det = EdgeDetector(image)
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

# HISTOGRAM GENERATION 
def histogram_process():
	img_list = [gray_image, prepim]
	labels_list = ['Grayscale Image','Preprocessed Image']
	hist_obj = Histogram("Grayscale - Pre-filtered")
	hist_obj.create_histogram(img_list, labels_list)

# SHAPE ANALSIS

def contour_list_manipulation(clist, median_val):
	contours = []
	areas = []
	ori_list = []
	initial_value = len(clist)
	for elem in clist:
		pts , area = elem
		if area >= median_val:
			contours.append(pts)
			areas.append(area)
			ori_list.append((pts, area))
		
	max_area_value = max(areas)

	for i, elem in enumerate(ori_list):
		pts, area = elem
		if area == max_area_value:
			ori_list.pop(i)
			break

	final_value = len(ori_list)
	print(f"The List has been reduced to {initial_value - final_value}")
	return ori_list

def median_value(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    if n % 2 == 1:
        median = sorted_lst[n // 2]
    else:
        median = (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2

    return median

def contour_fine_tuning(clist):
	areas = []
	for elem in clist:
		pts, area = elem
		areas.append(area)

	med_areas = median_value(areas)
	ori_list = contour_list_manipulation(clist, med_areas)
	return ori_list

def morphological_processing(image):
	print("Erosion Process happening")
	morphing = Morphological(kernel_size=5)
	openings = morphing.open(image, iterations=1)
	closings = morphing.close(openings, iterations=1)
	cv2.imwrite(os.path.join(output_path, "Opening_example.jpg"), closings)
 
 # Contour Processing

def trial_contour():
	image = cv2.imread(os.path.join(input_path, "trial_shapes.jpg"))

	object_cnt = ContourExtraction(approximation=cv2.CHAIN_APPROX_NONE)
	bin = object_cnt.preprocess(image)
	contours, heirarchy = object_cnt.extract(bin)
	for cnt in contours:
		for pt in cnt:
			x, y = pt[0]
			cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
	cv2.imwrite(os.path.join(output_path, "no_chain_approx.jpg"), image)

if __name__=='__main__':
	
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	pre_processed_img = PreProcessor(image)
	pre_processed_img.gaussian_blur()
	pre_processed_img.bilateral_denoise()
	prepim = pre_processed_img.histogram_equalization()
	cv2.imwrite(os.path.join(output_path, "pre_processed.jpg"), prepim)

	print("Start Edge Extraction")
	preprosimg = cv2.imread(os.path.join(output_path, "pre_processed.jpg"))

	cannyimg, sobelimg, scharrimg, prewittimg = edge_detection_process()
	# image_seg, prewitt_seg, canny_seg, sobel_seg, scharr_seg = segmenation_process()
	# morphing_process()

	edge_image_list = [cannyimg, sobelimg, scharrimg, prewittimg]



	# edged_img = cv2.imread(os.path.join(output_path, "canny.jpg"))

	contour_extraction_obj = ContourExtraction(approximation=cv2.CHAIN_APPROX_NONE)
	bin = contour_extraction_obj.preprocess(cannyimg)
	contours, binary = contour_extraction_obj.extract(bin)
	area_contour_list = contour_extraction_obj.shapeAreas(contours)	
	
	areas_list = []
	for elem in area_contour_list:
		pts, area = elem
		areas_list.append(pts)

	# new_contours_areas = contour_fine_tuning(area_contour_list)
	# newer_contour = contour_fine_tuning(new_contours)
	# morphological_processing(exp_image)

	# new_contours = []
	# for elem in new_contours_areas:
	# 	pts, areas = elem
	# 	new_contours.append(pts)

	canvas = contour_extraction_obj.draw(image, contours)
	cv2.imwrite(os.path.join(output_path, "Shape_analysis.jpg"), canvas)
	# canvas = contour_extraction_obj.fill(image, contours)
	# cv2.imwrite(os.path.join(output_path, "Filled_shapes.jpg"), canvas)

	shape_processing = BoundingBoxCreation(canvas, areas_list)
	reduced_image = shape_processing.createBoundingBoxes()

	cv2.imwrite(os.path.join(output_path, "ori_image.jpg"), reduced_image)
	# cv2.imwrite(os.path.join(output_path, "Reduced_Image.jpg"), reduced_image)
	# cv2.imwrite(os.path.join(output_path, "New_formatted_Image.jpg"), edge_image)


	for element in area_contour_list:
		pts, areas = element	
		print(f"The areas are: {areas}, and the cotour point list is: {pts}")
	# trial_contour()









	# histogram_obj = Histogram("Areas Distribution", "Areas", "Frequency")
	# histogram_obj.plot2histograms(areas_list, "Areas_Histogram.jpg", output_path, higher_end=500, bins=300)