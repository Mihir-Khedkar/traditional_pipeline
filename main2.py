from image_loading.loader import ImageRaw, ImageProcessed
from image_processing.pre_processing import PreProcessor
from image_processing.edge_detection import EdgeDetector
from image_processing.segmentation import Segmentation
from shape_descriptors.shape_representation import ContourAnalysis, BoundingBoxInfo
from utilities.histograms import Histogram
from utilities.configuration import Logger
from contour_extraction.countour_extraction import Morphological, ContourExtraction, BoundingBoxCreation
from shape_analysis.feature_extractor import HoughLineDetector

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# STATIC PATHS

input_path = r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\inputs'
output_path = r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\outputs'
logger = Logger()


def preprocessing_step(image):    
    pre_processed_img = PreProcessor(image)
    logger.minilogger("Applying Gaussian Blur")
    pre_processed_img.gaussian_blur()
    logger.minilogger("Applying Bilateral Denoise")
    pre_processed_img.bilateral_denoise()
    logger.minilogger("Applying Histogram Equalization")
    prepim = pre_processed_img.histogram_equalization()
	
    cv2.imwrite(os.path.join(output_path, "preprocessing_step.jpg"), prepim)
    logger.minilogger("preprocessing_step.jpg saved")

def segmentation_step(image):
    seg = Segmentation(image)
    logger.minilogger("Applying Otsu's Thresholding")
    segmented_image = seg.threshold_otsu()

    cv2.imwrite(os.path.join(output_path, "segmentation_step.jpg"), segmented_image)
    logger.minilogger("segmentation_step.jpg saved")

def edge_detection_step(image):
    edge_detector = EdgeDetector(image)
    logger.minilogger("Applying Canny Edge Detection with Scharr Gradient")
    logger.minilogger("Scharr Gradient Calculation")
    edge_detector.canny_scharr_gradient()
    logger.minilogger("Non-Maximum Suppression")
    edge_detector.canny_non_max_suppression(wk_multiplier=0.1, str_multiplier=0.15)
    logger.minilogger("Hysteresis Thresholding")
    edges = edge_detector.canny_hysteresis()

    cv2.imwrite(os.path.join(output_path, "edge_detection_step.jpg"), edges)
    logger.minilogger("edge_detection_step.jpg saved")

def contour_gerneration_step(image):
    contour_extraction_obj = ContourExtraction(approximation=cv2.CHAIN_APPROX_NONE)
    bin = contour_extraction_obj.preprocess(image)
    
    logger.minilogger("Extracting contours from the edged image")
    contours, binary = contour_extraction_obj.extract(bin)
    logger.minilogger(f"Extracted {len(contours)} contours from the edged image")

    return contours

def morphological_operator(image, operator, iterations=1):
    if operator == "dilate":
        logger.minilogger("Applying Dilation on the edged image")
        cont_morph = Morphological(kernel_size=5)
        edged_img = cont_morph.dilate(image, iterations)

    elif operator == "erode":
        logger.minilogger("Applying Erosion on the edged image")
        cont_morph = Morphological(kernel_size=5)
        edged_img = cont_morph.erode(image, iterations)

    elif operator == "open":
        logger.minilogger("Applying Opening on the edged image")
        cont_morph = Morphological(kernel_size=5)
        edged_img = cont_morph.open(image, iterations)
    
    elif operator == "close":
        logger.minilogger("Applying Closing on the edged image")
        cont_morph = Morphological(kernel_size=5)
        edged_img = cont_morph.close(image, iterations)

    elif operator == "gradient":
        logger.minilogger("Applying Gradient on the edged image")
        cont_morph = Morphological(kernel_size=5)
        edged_img = cont_morph.gradient(image, iterations)
    else:
        raise ValueError("Unsupported morphological operator. Choose from 'dilate', 'erode', 'open', 'close', 'gardient'.")

    return edged_img

def optimiser_operator(contours):
    area_list = []
    logger.minilogger(f"Initial number of contours: {len(contours)}")
    
    cnt_analysis = ContourAnalysis(contours)

    for contour in contours:
        feature = cnt_analysis.compute_features(contour)
        area_list.append(feature["area"])
	
    threshold_area = sum(area_list) / len(area_list)
    logger.minilogger(f"the threshold value is {threshold_area}")

    for index, contour in enumerate(contours):
        feature = cnt_analysis.compute_features(contour)
        if feature["area"] < threshold_area:
			# print(f"Removing Contour {index} with area {feature['area']}")
            removed_shape = contours.pop(index)
    
    logger.minilogger(f"Number of contours after optimisation: {len(contours)}")

    del cnt_analysis
    return contours

def contour_drawing_step(image, contours):
    contour_extraction_obj = ContourExtraction(approximation=cv2.CHAIN_APPROX_NONE)
    
    canvas_black = np.zeros_like(image)
    canvas = contour_extraction_obj.draw(canvas_black, contours)

    return canvas

def contour_fill_operator(image, contours):
    filler = ContourExtraction()
    bin_image = filler.preprocess(image)
    contours, _ = filler.extract(bin_image)
    filled_image = filler.fill(image, contours)
    logger.minilogger("Contours filled in the image")

    return filled_image

def heatmap_operator(image, name):

    h = Histogram(title=f"Contour Heatmap")
    file_name = f"heatmap_{name}"

    h.create_2d_patch_histogram(image, 
                                patch_size=(16,16), 
                                count_mode='nonzero', 
                                output_path=output_path, 
                                file_name=file_name)  

    logger.minilogger(f"Heatmap saved as: {file_name}")

def histogram_operator(data, title):
    h = Histogram(title=title)
    h.histogram_from_list(data, bins=100)


if __name__ == "__main__":
    
    logger.clog("Pipeline started")
    
    print("--------------------------------------------------------------------------------------------------------------------")
    
    # logger.clog("Loading image")
    # image = cv2.imread(os.path.join(input_path, "trial.jpg"))
    # preprocessing_step(image)
    # logger.clog("Pre-processing completed")
    
    print("--------------------------------------------------------------------------------------------------------------------")
    
    # logger.clog("Segmentation step started")
    # image = cv2.imread(os.path.join(output_path, "preprocessing_step.jpg"))
    # segmentation_step(image)
    # logger.clog("Segmentation completed")

    print("--------------------------------------------------------------------------------------------------------------------")

    # logger.clog("Edge detection step started")
    # image = cv2.imread(os.path.join(output_path, "segmentation_step.jpg"))
    # edge_detection_step(image)
    # logger.clog("Edge detection completed")

    print("--------------------------------------------------------------------------------------------------------------------")
    # logger.clog("Applying morphological operator: Dilate")
    # image = cv2.imread(os.path.join(output_path, "edge_detection_step.jpg"))
    # image_mrph = morphological_operator(image, operator="dilate")

    # cv2.imwrite(os.path.join(output_path, "morphological_operator_edge_detection_step.jpg"), image_mrph)

    print("--------------------------------------------------------------------------------------------------------------------")
    
    image = cv2.imread(os.path.join(output_path, "morphological_operator_edge_detection_step.jpg"))
    contours = contour_gerneration_step(image)
    original_number = len(contours)

    print("--------------------------------------------------------------------------------------------------------------------")
    
    logger.clog("Reducing the number of contours by setting mean as threshold")
    while True:
        print("------")
        new_contours = optimiser_operator(contours)
        contours = new_contours

        canvas = contour_drawing_step(image, contours)

# Applying a Dilation (follewed by erosion perhaps?) to the contours to join the shapes. 
        canvas = morphological_operator(canvas, operator="dilate", iterations=1)
        canvas = morphological_operator(canvas, operator="erode", iterations=1)

# Saving the iterative file as well as generating a heatmap        
        file_name = f'_{len(contours)}.jpg'
        heatmap_operator(canvas, file_name)
        cv2.imwrite(os.path.join(output_path, "contour_drawing_step" + file_name), canvas)
        logger.minilogger(f"contour_drawing_step{file_name} saved")
    
        if len(contours) < 50:
            cv2.imwrite(os.path.join(output_path, "final_contours_step.jpg"), canvas)
            break

    logger.minilogger(f"Contour optimisation completed. Contours maintained: {len(contours)}")
    logger.clog("Final contours saved as: final_contours_step.jpg" )

    print("--------------------------------------------------------------------------------------------------------------------")

# Applying Erode opertor to remove the weak connections
    logger.clog("Applying final morphological operations to strengthen contours")
    image = cv2.imread(os.path.join(output_path, "final_contours_step.jpg"))
    
    image2 = morphological_operator(image, operator="dilate", iterations=1)

    final_image = morphological_operator(image2, operator="gradient", iterations=1) 

    cv2.imwrite(os.path.join(output_path, "morphed_final_image.jpg"), final_image)

    print("--------------------------------------------------------------------------------------------------------------------")

# Hough Line Detection on the final morphed image
    logger.clog("Applying Hough Line Detection on the final morphed image")
    image = cv2.imread(os.path.join(output_path, "morphed_final_image.jpg"))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hough_detector = HoughLineDetector(gray_image)
    lines = hough_detector.detect_probabilistic(rho=1, theta=np.pi/180, threshold=80, min_line_length=30, max_line_gap=5)

    for line in lines:
        print(f"{line}")
        break