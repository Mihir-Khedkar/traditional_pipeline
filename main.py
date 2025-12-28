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



def edge_detection_process(image):
	edges_det = EdgeDetector(image)
	cannyimg  = edges_det.canny_edge()
	cv2.imwrite(os.path.join(output_path, "Canny_cv2.jpg"), cannyimg)
	
	# custom_cannyimg = EdgeDetector(image)
	# custom_cannyimg.canny_scharr_gradient()
	# custom_cannyimg.canny_non_max_suppression(wk_multiplier=0.1, str_multiplier=0.15)
	# edges = custom_cannyimg.canny_hysteresis()
	# cv2.imwrite(os.path.join(output_path, "Canny_w_Scharr.jpg"), edges)
	# sobelimg = edges_det.sobel()
	# cv2.imwrite(os.path.join(output_path, "sobel.jpg"), sobelimg)
	
	# scharrimgx = edges_det.scharr(dy=0)
	# scharrimgy = edges_det.scharr(dx=0)
	# scharrimg = (scharrimgx + scharrimgy)
	# cv2.imwrite(os.path.join(output_path, "scharr.jpg"), scharrimg)
	
	# prewittimg = edges_det.prewitt()
	# cv2.imwrite(os.path.join(output_path, "prewitt.jpg"), prewittimg)
	# return cannyimg, sobelimg, scharrimg, prewittimg
	

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

def area_list_modification(contours, edged_img):
	area_list = []

	cnt_analysis = ContourAnalysis(contours, edged_img)
	for contour in contours:
		feature = cnt_analysis.compute_features(contour)
		area_list.append(feature["area"])
	
	threshold_area = sum(area_list) / len(area_list)
	print(f"the threshold value is {threshold_area}")

	for index, contour in enumerate(contours):
		feature = cnt_analysis.compute_features(contour)
		if feature["area"] < threshold_area:
			# print(f"Removing Contour {index} with area {feature['area']}")
			removed_shape = contours.pop(index)
	
	del cnt_analysis
	return contours

if __name__=='__main__':
	
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	pre_processed_img = PreProcessor(image)
	pre_processed_img.gaussian_blur()
	pre_processed_img.bilateral_denoise()
	prepim = pre_processed_img.histogram_equalization()
	
	cv2.imwrite(os.path.join(output_path, "pre_processed.jpg"), prepim)
	# h = Histogram(title="Histograms of Grayscale and Pre-processed Images", xlabel="Intensity Value", ylabel="Pixel Count")
	# h.create_intensity_histogram(gray_image, header="Grayscale Image", bins=50, output_path=output_path, file_name="Grayscale_Histogram.jpg", show=False)
	# cv2.imwrite(os.path.join(output_path, "pre_processed.jpg"), prepim)
	# h.create_intensity_histogram(prepim, header="Preprocessed Image", bins=50, output_path=output_path, file_name="Preprocessed_Histogram.jpg", show=False)

# Segmentation
	preprosimg = cv2.imread(os.path.join(output_path, "pre_processed.jpg"))
	segmimg = Segmentation(preprosimg)
	otsu_img = segmimg.threshold_otsu()

	cv2.imwrite(os.path.join(output_path, "otsu_segmented.jpg"), otsu_img)

# # Morphological Operators
	seg_preprosimg = cv2.imread(os.path.join(output_path, "otsu_segmented.jpg"))
# 	morph = Morphological(kernel_size=5)
	
# 	morphed_image = morph.gradient(seg_preprosimg, iterations=1)
# 	cv2.imwrite(os.path.join(output_path, "gradient5f_iter1.jpg"), morphed_image)

# 	gradient_image = cv2.imread(os.path.join(output_path, "gradient_iter1.jpg"))
# 	opened_image = morph.erode(gradient_image, iterations=1)
# 	cv2.imwrite(os.path.join(output_path, "erode_grad_image5f.jpg"), opened_image)
# Edge Detection Process

	print("Start Edge Extraction using Normal Canny")

	edge_detection_process(seg_preprosimg)
	# cannyimg, sobelimg, scharrimg, prewittimg = edge_detection_process()
	# image_seg, prewitt_seg, canny_seg, sobel_seg, scharr_seg = segmenation_process()
	# morphing_process()

	# edge_image_list = [cannyimg, sobelimg, scharrimg, prewittimg]



	raw_edged_img = cv2.imread(os.path.join(output_path, "Canny_cv2.jpg"))
	# edged_img = cv2.imread(os.path.join(output_path, "dilate_grad_image.jpg"))

	cont_morph = Morphological(kernel_size=5)
	temp = cont_morph.dilate(raw_edged_img, iterations=2)
	edged_img = cont_morph.erode(temp, iterations=1)
	

	cv2.imwrite(os.path.join(output_path, "edged_morphed.jpg"), edged_img)

	contour_extraction_obj = ContourExtraction(approximation=cv2.CHAIN_APPROX_NONE)
	bin = contour_extraction_obj.preprocess(edged_img)
	contours, binary = contour_extraction_obj.extract(bin)
	area_contour_list = contour_extraction_obj.shapeAreas(contours)	
	print(f"extracted contours and computed areas {len(area_contour_list)}")
	# new_contours_areas = contour_fine_tuning(area_contour_list)
	# newer_contour = contour_fine_tuning(new_contours)
	# morphological_processing(exp_image)

	for i in range(5):
		new_contours = area_list_modification(contours, edged_img)
		contours = new_contours

	# new_contours = []
	# for elem in new_contours_areas:
	# 	pts, areas = elem
	# 	new_contours.append(pts)

	canvas_black = np.zeros_like(image)

	canvas = contour_extraction_obj.draw(canvas_black, contours)

	print(f"Saving contour images to {output_path}")
	cv2.imwrite(os.path.join(output_path, "Shape_contours.jpg"), canvas)

	h1 = Histogram(title="Pre-Contour Removal")
	h1.create_2d_patch_histogram(canvas)

	# h3 = Histogram(title="Contour Area Analysis")
	# h3.histogram_from_list(area_list, bins='auto')

	pre_cnt = len(contours)
	post_cnt = len(contours)
	print(f"Contours before removal: {pre_cnt}, Contours after removal: {post_cnt}")

	canvas = contour_extraction_obj.draw(canvas_black, contours)
	canvas_filled = contour_extraction_obj.fill(canvas_black, contours)

	h2 = Histogram(title="Post-Contour Removal")
	h2.create_2d_patch_histogram(canvas)

	print(f"Saving contour images to {output_path}")
	cv2.imwrite(os.path.join(output_path, "Shape_contours.jpg"), canvas)

	# canvas = contour_extraction_obj.fill(image, contours)
	# cv2.imwrite(os.path.join(output_path, "Filled_shapes.jpg"), canvas)

	
	# areas_list = []
	# for elem in area_contour_list:
	# 	pts, area = elem
	# 	areas_list.append(pts)

	shape_processing = BoundingBoxCreation(image, areas_list)
	# reduced_image = shape_processing.createBoundingBoxes()

	# cv2.imwrite(os.path.join(output_path, "bounding_boxes.jpg"), reduced_image)
	# cv2.imwrite(os.path.join(output_path, "Reduced_Image.jpg"), reduced_image)
	# cv2.imwrite(os.path.join(output_path, "New_formatted_Image.jpg"), edge_image)


	# for element in area_contour_list:
	# 	pts, areas = element	
	# 	print(f"The areas are: {areas}, and the cotour point list is: {pts}")
	# trial_contour()
	
	# for bbox in elimination_list:
	# 	x, y, w, h = bbox
	# 	edged_img[y:y+h,x:x+w] = (0,0,0)

	# cv2.imwrite(os.path.join(output_path, "Reduced Image.jpg"), edged_img)
	# classified_image = cnt_analysis.visualize()
	# cv2.imwrite(os.path.join(output_path, "Classified_Contours.jpg"), classified_image)