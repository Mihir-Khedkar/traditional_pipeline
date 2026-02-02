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
import math
import colorsys


# █▀█ ▄▀█ ▀█▀ █░█ █▀
# █▀▀ █▀█ ░█░ █▀█ ▄█

input_path = r'inputs'
output_path = r'outputs'
logger = Logger()


# █▀ ▀█▀ ▄▀█ ▀█▀ █ █▀▀   █░█ ▄▀█ █▀█ █ ▄▀█ █▄▄ █░░ █▀▀ █▀
# ▄█ ░█░ █▀█ ░█░ █ █▄▄   ▀▄▀ █▀█ █▀▄ █ █▀█ █▄█ █▄▄ ██▄ ▄█

FILENAME = r"trial_small.jpg"
raw_image = cv2.imread(os.path.join(input_path, FILENAME))
H, W ,C = raw_image.shape

SEGMENTED_REGIONS = 4

UPPER_THRESHOLD = 300
LOWER_THRESHOLD = int(0.44*UPPER_THRESHOLD)

THRESHOLD_PROBAB = int(0.3*min(H,W))
MIN_LINE_LENGTH = 0.05*W
MAX_LINE_GAP = 10

THRESHOLD_STANDARD = int(0.3*min(H,W))

BIN_SELECTION_FILTER = 2
CONTOUR_NUMBER_FILTER = 100
DISTANCE_MARGIN = 10

REFINED_THRESHOLD_PARAMETER = 200

# █▀▄▀█ █▀▀ ▀█▀ █░█ █▀█ █▀▄ █▀
# █░▀░█ ██▄ ░█░ █▀█ █▄█ █▄▀ ▄█



def preprocessing_step(image):    
    pre_processed_img = PreProcessor(image)

    logger.minilogger("Applying Median Blur")
    pre_processed_img.median_blur()

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
    segmented_image = seg.multi_threshold_otsu(regions = SEGMENTED_REGIONS)

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

    # edges = edge_detector.canny_edge()
    # cv2.imwrite(os.path.join(output_path, "edge_detection_step_cv2canny.jpg"), edges)

    # logger.minilogger("edge_detection_step_cv2canny.jpg saved")
    logger.minilogger("Creating Heatmap of the edge image")
    h_edge = Histogram(title="Edge_heatmap", label="Heatmap - Canny Edge Detection")
    h_edge.create_2d_patch_histogram(edges, file_name="edge_heatmap.png")

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

def optimiser_operator(contours, aspect_ratio_threshold=0.5):
    area_list = []
    # logger.minilogger(f"Initial number of contours: {len(contours)}")
    
    cnt_analysis = ContourAnalysis(contours)

    for contour in contours:
        feature = cnt_analysis.compute_features(contour)
        area_list.append(feature["area"])
	
    threshold_area = sum(area_list) / len(area_list)
    # logger.minilogger(f"the threshold value is {threshold_area}")

    for index, contour in enumerate(contours):
        feature = cnt_analysis.compute_features(contour)
        if feature["area"] < threshold_area or feature["aspect_ratio"] < aspect_ratio_threshold:
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

    # logger.minilogger("Contours filled in the image")
    return filled_image

def heatmap_operator(image, name):

    h = Histogram(title=f"Contour Heatmap")
    file_name = f"heatmap_{name}"

    h.create_2d_patch_histogram(image, 
                                patch_size=(16,16), 
                                count_mode='nonzero', 
                                output_path=output_path, 
                                file_name=file_name)  

    # logger.minilogger(f"Heatmap saved as: {file_name}")

def histogram_operator(data, title):
    h = Histogram(title=title)
    h.histogram_from_list(data, bins=100)

def feature_detection_step(image, iteration_method=False):
    image = cv2.imread(os.path.join(output_path, "morphed_final_image.jpg"))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if iteration_method:
        thresholds = list(range(50, 501, 50))
        gaps = list(range(0,51,5))
        
        m = max(gray_image.shape)
        m2 = [0.0, 0.1, 0.2, 0.3]
        lengths = [int(item * m) for item in m2]

        lines_vals = []
        line_maxing = []

        hough_detector = HoughLineDetector(gray_image)
        for threshold in thresholds:
            for gap in gaps:
                for length in lengths:
                    lines = hough_detector.detect_probabilistic(rho=5.0, theta=np.pi/180, threshold=threshold, min_line_length=length, max_line_gap=gap)
                    if lines is None:
                        logger.minilogger(f"Hough_Probab_Transform_{threshold}_{gap}_{length} has 0 lines")
                        continue
                    # output = hough_detector.draw_probabilistic(image, lines)
                    filename = f"Hough_Probab_Transform_{threshold}_{gap}_{length}.jpg"
                    # cv2.imwrite(os.path.join(output_path, filename), output)
                    logger.minilogger(f"File: {filename} has {lines.shape[0]} lines")
                    lines_vals.append((threshold, gap, length, lines.shape[0]))
                    line_maxing.append(lines.shape[0])
        
        for line in line_maxing:
            max_value = max(line_maxing)
            for tline in lines_vals:
                if tline[3] == max_value:
                        if tline[0] == 0 or tline[2] == 0:
                            lines_vals.remove(tline)
                            line_maxing.remove(max_value)
                            continue
                        best_threshold = tline[0]
                        best_gap = tline[1]
                        best_length = tline[2]        
        
        logger.minilogger(f"Best Threshold: {best_threshold}, Best Gap: {best_gap}, Best Length: {best_length}, Max Lines: {max_value}")
    
    hough_detector = HoughLineDetector(gray_image)
    lines = hough_detector.detect_probabilistic(rho=5.0, theta=np.pi/180, threshold=200, min_line_length=867, max_line_gap=50)
    output = hough_detector.draw_probabilistic(image, lines)
    n, x, y = lines.shape
    logger.minilogger(f"Number of lines detected: {n}")
    cv2.imwrite(os.path.join(output_path, "hough_lines_detected.jpg"), output)
    logger.minilogger("hough_lines_detected.jpg saved")

    return lines

def slope_calc_operator(line):
    x1, y1, x2, y2 = line
    
    if (x1 - x2) == 0:
        angle_rad = np.pi / 2
    else:
        m = (y1 -y2)/(x1 - x2)
        angle_rad = math.atan(m)

    angle_deg = math.degrees(angle_rad)

    if angle_deg < 0:
        angle_deg += 180
    
    return angle_deg

def angular_distance(a, b):
    """Smallest distance between angles in [0, π)."""
    d = abs(a - b)
    return min(d, np.pi - d)


def select_hough_lines_by_houghP(
    hough_lines,        # [(rho, theta)]
    houghP_lines       # [(x1, y1, x2, y2)]
):
    selected = []
    
    for plines in houghP_lines:
        x1, y1, x2, y2 = plines
        dx = x1 - x2
        dy = y1 - y2
        
        A = dy
        B = -dx
        C = dx * y1 - dy * x1

        d = abs(C) / math.sqrt(A**2 + B**2)

        angle_rad = math.atan2(dy, dx)

        for hline in hough_lines:
            rho, theta = hline
            theta += np.pi/2
            angle_diff = abs(angle_rad - theta)

            distance = abs(d - rho)

            if angle_diff <= np.pi/180 and distance <= 5.0:
                if angle_diff < 0.0:
                    angle_diff += np.pi

                selected.append(hline)

                # logger.minilogger(f"Probab Line pts: {plines}, rho: {d}, theta: {math.degrees(angle_rad):.2f}, angle_diff: {angle_diff:.2f}, distance: {distance:.2f}")
                # logger.minilogger(f"Standard Line: rho: {rho}, theta: {math.degrees(theta):.2f}")
    
    return selected
 

def apply_median_filter(image, kernel_size=5):
    filtered_image = cv2.medianBlur(image, ksize=kernel_size)
    return filtered_image


def conversion_to_cartesian(lines, shape):
    h, w = shape
    cart_lines = []

    for pt in lines:
        rho, theta = pt[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + w * (-b))
        y1 = int(y0 + h * (a))
        x2 = int(x0 - w * (-b))
        y2 = int(y0 - h * (a))

        cart_lines.append((x1,y1,x2,y2))
    return cart_lines

def line_color_bgr(i, saturation=0.9, value=0.9):
    golden_ratio = 0.618033988749895
    hue = (i * golden_ratio) % 1.0

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

    return (
        int(b * 255),
        int(g * 255),
        int(r * 255)
    )

def point_in_rotated_box(point, box):
    return cv2.pointPolygonTest(box, tuple(point), False) >= 0

def segments_intersect(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and \
           (ccw(p1, p2, q1) != ccw(p1, p2, q2))

def line_intersects_rotated_box(p1, p2, rect):
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)

    # Case 1: endpoint inside box
    if point_in_rotated_box(p1, box) or point_in_rotated_box(p2, box):
        return True

    # Case 2: line intersects any box edge
    for i in range(4):
        q1 = box[i]
        q2 = box[(i + 1) % 4]
        if segments_intersect(p1, p2, q1, q2):
            return True

    return False

def calculate_orientation(point):
    x1, y1, x2, y2 = point
    if (x2-x1) == 0:
        m = 0
    else:
        m = (y2-y1)/(x2-x1)
    angle_rad = math.atan(m)
    angle_deg = math.degrees(angle_rad)

    if angle_deg < 0:
        angle_deg += 180
    
    return angle_deg

def calculate_length(point):
    x1, y1, x2, y2 = point
    length = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return length












# █▀█ █▀█ █▀▀ ▄▄ █▀█ █▀█ █▀█ █▀▀ █▀▀ █▀ █▀ █ █▄░█ █▀▀
# █▀▀ █▀▄ ██▄ ░░ █▀▀ █▀▄ █▄█ █▄▄ ██▄ ▄█ ▄█ █ █░▀█ █▄█


def preprocessing_module():
    logger.clog("Pre-processing step started")
    logger.minilogger("Loading image")

    image = cv2.imread(os.path.join(input_path, FILENAME))
    preprocessing_step(image)
    
    logger.clog("Pre-processing completed")
    
    logger.clog("Segmentation step started")
    
    image = cv2.imread(os.path.join(output_path, "preprocessing_step.jpg"))
    segmentation_step(image)
    
    logger.clog("Segmentation completed")




# █▀▀ █▀▄ █▀▀ █▀▀   █▀▄ █▀▀ ▀█▀ █▀▀ █▀▀ ▀█▀ █ █▀█ █▄░█
# ██▄ █▄▀ █▄█ ██▄   █▄▀ ██▄ ░█░ ██▄ █▄▄ ░█░ █ █▄█ █░▀█


def edge_detection_module():    
    logger.clog("Edge detection step started")
    
    image = cv2.imread(os.path.join(output_path, "segmentation_step.jpg"))
    edge_detection_step(image)
    
    logger.clog("Edge detection completed")

    logger.clog("Applying morphological operator: Close")
    
    image = cv2.imread(os.path.join(output_path, "edge_detection_step.jpg"))
    image_mrph = morphological_operator(image, operator="close", iterations=2)
    cv2.imwrite(os.path.join(output_path, "morphological_operator_edge_detection_step.jpg"), image_mrph)




# █▀▀ █▀█ █▄░█ ▀█▀ █▀█ █░█ █▀█   █▀▀ ▀▄▀ ▀█▀ █▀█ ▄▀█ █▀▀ ▀█▀ █ █▀█ █▄░█
# █▄▄ █▄█ █░▀█ ░█░ █▄█ █▄█ █▀▄   ██▄ █░█ ░█░ █▀▄ █▀█ █▄▄ ░█░ █ █▄█ █░▀█

def contour_extraction_module():
    logger.clog("Contour extraction and optimisation step started")

    image = cv2.imread(os.path.join(output_path, "morphological_operator_edge_detection_step.jpg"))
    
    logger.minilogger("Applying median filter")

    contours = contour_gerneration_step(image)
    original_number = len(contours)

    logger.minilogger(f"Reducing the number of contours:({original_number}) by setting mean as threshold")
    
    global reduced_contours
    switch = 0
    while switch < 10:
        # logger.minilogger("Optimising contours")
        # logger.minilogger(f"Current number of contours: {len(contours)}")
        
        new_contours = optimiser_operator(contours)
        contours = new_contours

        # logger.minilogger(f"Contours after optimisation: {len(contours)}")
        # logger.minilogger("Drawing contours on blank canvas")

        canvas = contour_drawing_step(image, contours)

        logger.minilogger("Applying morphological operations to strengthen contours - dilate")

        canvas = morphological_operator(canvas, operator="dilate", iterations=1)
        # canvas = morphological_operator(canvas, operator="open", iterations=1)
        file_name = f'_{len(contours)}.jpg'
        heatmap_operator(canvas, file_name)
        cv2.imwrite(os.path.join(output_path, "contour_drawing_step" + file_name), canvas)
        
        # logger.minilogger(f"contour_drawing_step{file_name} saved")
        switch += 1
        if len(contours) < CONTOUR_NUMBER_FILTER:
            reduced_contours = contours
            cv2.imwrite(os.path.join(output_path, "final_contours_step.jpg"), canvas)
            break
        
        

    logger.minilogger(f"Contour optimisation completed. Contours maintained: {len(contours)}")
    logger.minilogger("Final contours saved as: final_contours_step.jpg" )

    logger.minilogger("Applying final morphological operations to strengthen contours")
    
    image = cv2.imread(os.path.join(output_path, "final_contours_step.jpg"))
    final_image = morphological_operator(image, operator="close", iterations=2)
    # final_image = morphological_operator(image2, operator="gradient", iterations=1) 
    cv2.imwrite(os.path.join(output_path, "morphed_final_image.jpg"), final_image)

    logger.clog("Contour extraction and optimisation completed")

def contour_refinement_module():
    logger.clog("Starting Contour refinement module")
    
    image = cv2.imread(os.path.join(output_path, "final_contours_step.jpg"))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canvas = np.zeros_like(image)

    logger.minilogger("Applying Line Segment Detector")
    lsd = cv2.createLineSegmentDetector(
        refine=cv2.LSD_REFINE_STD
    )


    lines, widths, precisions, _ = lsd.detect(gray_image)

    img_color = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    distances = []
    pts = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            distances.append(distance)
            pts.append((x1, x2, y1, y2))
            cv2.line(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)

    mean = 0.9*max(distances)
    logger.minilogger(f"Distance Threshold: {mean}")
  
    angles = []
    for pt in pts:
        angle = calculate_orientation(pt)
        angles.append(angle)
    
    counts , bins = np.histogram(angles, bins=100)
    max_frequency = 0.75*max(counts)
    logger.minilogger(f"Angle Threshold frequency: {max_frequency}")

    limits = []
    for i, hist in enumerate(zip(bins, counts)):
        bin, count = hist
        if count > max_frequency:
            if i == 0:
                lower = 0
                upper = bin
            else:
                lower = bins[i-1]
                upper = bin
            logger.minilogger(f"Orientation bins chosen are: {lower} and {upper}")
            limits.append((lower, upper))

    canvas = np.zeros_like(img_color)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            for limit in limits:
                lower, upper = limit
                angle = calculate_orientation((x1, x2, y1, y2))
                length = calculate_length((x1, x2, y1, y2))
                if angle >= lower and angle <= upper:
                    if length >= mean:
                        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_path, "contour_refined_segments.jpg") , img_color)
    cv2.imwrite(os.path.join(output_path, "contour_refined_step.jpg") , canvas)


# █▀▀ █▀▀ ▄▀█ ▀█▀ █░█ █▀█ █▀▀   █▀▀ ▀▄▀ ▀█▀ █▀█ ▄▀█ █▀▀ ▀█▀ █ █▀█ █▄░█
# █▀░ ██▄ █▀█ ░█░ █▄█ █▀▄ ██▄   ██▄ █░█ ░█░ █▀▄ █▀█ █▄▄ ░█░ █ █▄█ █░▀█


def feature_extraction_module():
    logger.clog("Applying Hough Line Detection on the final morphed image")
    
    image = cv2.imread(os.path.join(output_path, "final_contours_step.jpg"), cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread(os.path.join(input_path, "final.jpg"), cv2.IMREAD_GRAYSCALE)
    ori_image1 = cv2.imread(os.path.join(input_path, FILENAME))
    ori_image2 = cv2.imread(os.path.join(input_path, FILENAME))
    canvas = np.zeros_like(image)

    logger.minilogger("Detecting lines using Standard Hough Transform and Probabilistic Hough Transform")

    hough_obj = HoughLineDetector(image)
    standard_lines = hough_obj.detect_standard(rho=1.0, theta=np.pi/180, threshold=THRESHOLD_PROBAB)
    probabilistic_lines = hough_obj.detect_probabilistic(rho=1.0, theta=np.pi/180, threshold=THRESHOLD_PROBAB, min_line_length=MIN_LINE_LENGTH, max_line_gap=MAX_LINE_GAP)

    if probabilistic_lines is None:
        print("Unable to detect lines. Reduce Parameters.")
    logger.minilogger(f"Number of Probabilistic Hough lines detected: {probabilistic_lines.shape[0]}")

    slopes = []
    probabilistic_points = []
    for line in probabilistic_lines:
        probabilistic_slopes = slope_calc_operator(line[0])
        slopes.append(probabilistic_slopes)
        probabilistic_points.append(line[0])

    logger.minilogger(f"Number of slope angles extracted: {len(slopes)} with minimum {min(slopes)} and maximum {max(slopes)}")
    logger.minilogger("Starting the Histogram calculation to classify line angles")
    counts, bin_edges = np.histogram(slopes, bins=180, range=(0, 180))

    counts_list = counts.tolist()
    bins_list = counts.tolist()

    global angle_maxing
    angle_maxing = []
    
    for i in range(BIN_SELECTION_FILTER):
        max_value = max(counts_list)
        idx = counts_list.index(max_value)
        
        counts_list[idx] = 0

        min_angle = bin_edges[idx]
        max_angle = bin_edges[idx+1]

        angle_maxing.append((min_angle, max_angle))

    logger.minilogger("Reduction of Probability Points")

    global reduced_probability_points
    reduced_probability_points = []
    for point in probabilistic_points:
        slope = slope_calc_operator(point)
        for angle in angle_maxing:
            mini, maxi = angle
            if slope <= maxi and slope >= mini:
                reduced_probability_points.append(point)
                break
            else:
                continue

    logger.minilogger(f"Reduced probabilistic list: {len(reduced_probability_points)}")
    logger.minilogger(f"Number of Standard Hough lines detected: {standard_lines.shape[0]}")
    

    # Drawing part

    for pt in reduced_probability_points:
        x1, y1, x2, y2 = pt 
        cv2.line(ori_image1, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.line(canvas, (x1, y1), (x2, y2), (255,255,255), 2)

    cv2.imwrite(os.path.join(output_path, "hough_lines_detected_black_canvas.jpg"), canvas)
    cv2.imwrite(os.path.join(output_path, "hough_lines_detected_modified_plus_probab.jpg"), ori_image1)
    logger.clog("Hough Line Detection completed and saved as: hough_lines_detected_modified_plus_probab.jpg")

    # cv2.imwrite(os.path.join(output_path, "hough_lines_detected_modified_plus_standard.jpg"), ori_image2)
    # logger.clog("Hough Line Detection completed and saved as: hough_lines_detected_modified_plus_standard.jpg")
    
    hough_obj_standard = HoughLineDetector(canvas)
    standard_lines = hough_obj_standard.detect_standard(rho=1.0, theta=np.pi/180, threshold=THRESHOLD_PROBAB)
    
    if standard_lines is not None:
        standard_lines = conversion_to_cartesian(standard_lines, ori_image1.shape[:2])
    else:
        logger.clog("No standard lines detected.")
        return

    std_lines_reduced_ang = []
    tmp = []
    for line in standard_lines:
        theta_d = slope_calc_operator(line)
        
        for angles in angle_maxing:
            mini, maxi = angles

            if theta_d <= maxi and theta_d >= mini:
                std_lines_reduced_ang.append(line)
                tmp.append(theta_d)            

    if len(tmp) > 0: 
        logger.minilogger(f"Standard Lines after angular elimination: {len(std_lines_reduced_ang)}, min value: {min(tmp)}, max value: {max(tmp)}")
    else:
        logger.minilogger(f"No standard line detected")
    del tmp

    standard_lines_reduced = []

    for line in std_lines_reduced_ang:
        x1, y1, x2, y2 = line
        
        dx = x1 - x2
        dy = y1 - y2

        if dx != 0 and dy !=0:
            a = 1/dx
            b = -1/dy 
            c = y1/dy - x1/dx

        else:
            a = 0
            b = 0
            c = 0

        for pline in reduced_probability_points:
            if a == 0 or b == 0:
                d1 = 1000
                d2 = 1000
            else:
                xp1, yp1, xp2, yp2 = pline
                d1 = (a*xp1 + b*yp1 + c)/np.sqrt(a**2 + b**2)
                d2 = (a*xp2 + b*yp2 + c)/np.sqrt(a**2 + b**2)

            if abs(d1) < DISTANCE_MARGIN and abs(d2) < DISTANCE_MARGIN:
                standard_lines_reduced.append(line)
                break
            else:
                continue

    logger.minilogger(f"Standard line after proximity reduction: {len(standard_lines_reduced)}")

    for pt in standard_lines_reduced:
        x1, y1, x2, y2 = pt
        cv2.line(ori_image2, (x1, y1), (x2, y2), (0,0,255), 2)

    cv2.imwrite(os.path.join(output_path, "hough_lines_detected_modified_plus_standard.jpg"), ori_image2)
    logger.clog("Hough Line Detection completed and saved as: hough_lines_detected_modified_plus_standard.jpg")

def feature_extraction_step_extra():
    image = cv2.imread(os.path.join(output_path, "contour_refined_step.jpg"))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ori_image = cv2.imread(os.path.join(output_path, "hough_lines_detected_modified_plus_standard.jpg"))

    hough_obj = HoughLineDetector(gray_image)
    standard_lines = hough_obj.detect_standard(rho=2.0, theta=np.pi/180, threshold=REFINED_THRESHOLD_PARAMETER)
    
    standard_lines = conversion_to_cartesian(standard_lines, ori_image.shape[:2])
    
    std_lines_reduced_ang = []
    tmp = []
    for line in standard_lines:
        theta_d = slope_calc_operator(line)
        
        for angles in angle_maxing:
            mini, maxi = angles

            if theta_d <= maxi and theta_d >= mini:
                std_lines_reduced_ang.append(line)
                tmp.append(theta_d)            

    if len(tmp) > 0: 
        logger.minilogger(f"Standard Lines after angular elimination: {len(std_lines_reduced_ang)}, min value: {min(tmp)}, max value: {max(tmp)}")
    else:
        logger.minilogger(f"No standard line detected")
    del tmp

    standard_lines_reduced = []

    for line in std_lines_reduced_ang:
        x1, y1, x2, y2 = line
        
        dx = x1 - x2
        dy = y1 - y2

        if dx != 0 and dy !=0:
            a = 1/dx
            b = -1/dy 
            c = y1/dy - x1/dx

        else:
            a = 0
            b = 0
            c = 0

        for pline in reduced_probability_points:
            if a == 0 or b == 0:
                d1 = 1000
                d2 = 1000
            else:
                xp1, yp1, xp2, yp2 = pline
                d1 = (a*xp1 + b*yp1 + c)/np.sqrt(a**2 + b**2)
                d2 = (a*xp2 + b*yp2 + c)/np.sqrt(a**2 + b**2)

            if abs(d1) < DISTANCE_MARGIN and abs(d2) < DISTANCE_MARGIN:
                standard_lines_reduced.append(line)
                break
            else:
                continue

    logger.minilogger(f"Standard line after proximity reduction: {len(standard_lines_reduced)}")

    for pt in standard_lines_reduced:
        x1, y1, x2, y2 = pt
        cv2.line(ori_image, (x1, y1), (x2, y2), (0,0,255), 2)

    cv2.imwrite(os.path.join(output_path, "Standard_line_detection_refined.jpg"), ori_image)
    # if standard_lines is not None:
    #     drawing_std = hough_obj.draw_standard(ori_image, standard_lines)
        
    # else:
    #     print("No lines were Detected")

# █▀█ █▀▀ █▀ █░█ █░░ ▀█▀   █▀▀ ▄▀█ █░░ █▀▀ █░█ █░░ ▄▀█ ▀█▀ █ █▀█ █▄░█ █▀
# █▀▄ ██▄ ▄█ █▄█ █▄▄ ░█░   █▄▄ █▀█ █▄▄ █▄▄ █▄█ █▄▄ █▀█ ░█░ █ █▄█ █░▀█ ▄█


def result_calculation_module():
    image = cv2.imread(os.path.join(output_path, "hough_lines_detected_modified_plus_standard.jpg"))
    # image2 = cv2.imread(os.path.join(output_path, "hough_lines_detected_modified_plus_standard.jpg"))

    logger.clog("Creating Bounding Boxes")
    # logger.clog("Creating Bounding Boxes - tight")

    thick = 10
    bb_coord = []
    for line in reduced_probability_points:
        x1, y1, x2, y2= line
        
        mnx = min(x1, x2) - thick    
        mny = min(y1, y2) - thick

        mxx = max(x1, x2) + thick
        mxy = max(y1, y2) + thick

        # w = abs(mnx - mxx) + 25
        # h = abs(mny - mxy) + 25
        # bb_coord.append((mnx, mny, w, h))
        bb_coord.append((mnx, mny, mxx, mxy))
        
        

    for i in range(len(bb_coord)):
        x1, y1, x2, y2 = bb_coord[i]
        color = line_color_bgr(i)
        cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)

    # for pt in reduced_probability_points:
    #     x1, y1, x2, y2= pt
    #     p1 = (x1, y1)
    #     p2 = (x2, y2)

    #     for i, cnt in enumerate(reduced_contours):
    #         rect = cv2.minAreaRect(cnt)
    #         box = cv2.boxPoints(rect)
    #         box = np.array(box, dtype=np.float32)
    #         if line_intersects_rotated_box(p1, p2, rect):
    #             box = np.int0(box)
    #             color = line_color_bgr(i)
    #             cv2.drawContours(image2, [box], 0, color, 2)
    #         else:
    #             continue
        
				
    cv2.imwrite(os.path.join(output_path, "BB_image.jpg"), image)
    # cv2.imwrite(os.path.join(output_path, "BB_image_tight.jpg"), image2)
    logger.clog('Successfully created Bounding Box image - normal')




# ░█▀▄▀█ ─█▀▀█ ▀█▀ ░█▄─░█ 
# ░█░█░█ ░█▄▄█ ░█─ ░█░█░█ 
# ░█──░█ ░█─░█ ▄█▄ ░█──▀█


if __name__ == "__main__":
    
    logger.clog("Pipeline started")
    logger.clog(f"Image filename: {FILENAME} with shape H:{H}, W:{W}, C:{C}")
    
    preprocessing_module()
    edge_detection_module()
    contour_extraction_module()
    contour_refinement_module()
    feature_extraction_module()
    feature_extraction_step_extra()
    result_calculation_module()

    logger.clog("Pipeline completed")