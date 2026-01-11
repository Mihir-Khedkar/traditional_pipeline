import numpy as np
import cv2
import os


input_path = r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\inputs'
output_path = r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\outputs'

image = cv2.imread(os.path.join(input_path, "final.jpg"), cv2.IMREAD_GRAYSCALE)

h, w = image.shape

non_zero_ind = np.nonzero(image)

image[non_zero_ind] = 255
median_filtered_img = cv2.medianBlur(image, 5)
edges = cv2.Canny(image, 100, 250, L2gradient=True)

# Display the original and filtered images
# cv2.imshow('Median Filtered Image (5x5 kernel)', median_filtered_img)

# cv2.imwrite(os.path.join(output_path, "dark_on_white.jpg"), image)

contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

canvas = np.zeros_like(image)
# cv2.drawContours(canvas, contours, contourIdx=-1, color=(255, 255, 255), thickness=2)
canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
def is_straight(contour, max_error=2.0):
    # Fit line
    vx, vy, x0, y0 = cv2.fitLine(
        contour, cv2.DIST_L2, 0, 0.01, 0.01
    )

    # Direction vector
    direction = np.array([vx, vy]).reshape(2)
    point_on_line = np.array([x0, y0]).reshape(2)

    # Compute perpendicular distance of each point
    pts = contour.reshape(-1, 2)
    diff = pts - point_on_line
    dist = np.abs(np.cross(diff, direction)) / np.linalg.norm(direction)

    return np.mean(dist) < max_error

straight_contours = [
    c for c in contours if len(c) > 55 and is_straight(c)
]

cv2.drawContours(canvas, straight_contours, -1, (0, 255, 0), 2)

cv2.imshow('Contours', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
