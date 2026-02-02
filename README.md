# Mapping of Ship Surfaces using Classical Computer Vision algorithms

This repository contains a pipeline to extract structural elements from a given image of a ship surface. It does that by extracting the edges from the image and trying to find the straight edges within that edge image.


##  Overview

This project implements a modular pipeline consisting of:
- Pre-processing
- Edge Extraction
- Contour Extraction and Analysis
- Feature Extraction


The objective is to process raw visual data, identify relevant structural elements, and generate a usable semantic map for robotic applications.

The current end output are multiple Bounding Boxes of the identified straight lines.

COMING SOON: Consolidated Bounding Boxes

---



## Installing dependencies using pip3

Dependencies can be installed from the requirement.txt. 
(Its good practise to create a Virtual Environment to install the dependencies)

pip install -r requirements.txt

---



## Accessing the Code

This code was built on 3.12.3
Use the main.py file, in the parent directory to access the main code.
The modules can be found in the bottom of the code in the __main__ class.

## Using the Code 

The image you want to work on should be written in the STATIC VARIABLES section (~ line 26) under the FILENAME variable.
Please ensure that the image is placed in the input directory.

---


## Adjusting the Parameters 

The following Parameters have been defined in the STATIC VARIABLES section:

UPPER_THRESHOLD = 200 #int value. Threshold for Canny Edge Detection. Decides which edges to keep and which to discard.

# Check OpenCV documentation: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
THRESHOLD_PROBAB = 500 # int value.
MIN_LINE_LENGTH = 578 # int value. 
MAX_LINE_GAP = 45 # int value.

# Check OpenCV documentation: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
THRESHOLD_STANDARD = 250

# Denoising Parameters

    BIN_SELECTION_FILTER = 2 # int value. Orientation bins to retain. eg. 179 to 180 degrees orientation can be an orientation bin. Lines retained will from the orientation bin.

    CONTOUR_NUMBER_FILTER = 50 # int value. Number of shapes to retain after denoising. Smaller the number, lesser the features in the image. It eliminates shapes based on smaller areas.

    DISTANCE_MARGIN = 10 # int value. We first calculate the Probablistic Hough Transform lines and then the Standard lines using them. This specifies the minimum distance from the Probabilistic Hough Transform 


## IMPORTANT

Any changes should be made in a cloned branch and not directly in the main branch. Although I have written the code, only GOD will know how to fix it incase something breaks.
