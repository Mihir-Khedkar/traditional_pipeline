import cv2


class EdgeDetector:
	def __init__(self, image):
		self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
	def canny_edge(self, lower_threshold=100, upper_threshold=200):
		edges = cv2.Canny(self.image, lower_threshold, upper_threshold)
		return edges
	
	def sobel(self, dx=1, dy=1, ksize=3):
		sobel = cv2.Sobel(self.image, cv2.CV_64F, dx, dy, ksize=ksize)
		return cv2.convertScaleAbs(sobel)
		
	def scharr(self, dx=1, dy=1):
		scharr = cv2.Scharr(self.image, cv2.CV_64F, dx, dy)
		return cv2.convertScaleAbs(scharr)
		
	def prewitt(self):
		import numpy as np
	
		kernel_x = np.array([[1,0,-1],
							[1,0,-1],
							[1,0,-1]], dtype=np.float32)
							
		
		kernel_y = np.array([[1,1,1],
							[0,0,0],
							[-1,-1,-1]], dtype=np.float32)
		
		prewitt_x = cv2.filter2D(self.image, -1, kernel_x)
		prewitt_y = cv2.filter2D(self.image, -1, kernel_y)
		
		prewitt = cv2.magnitude(prewitt_x.astype(np.float32),
								prewitt_y.astype(np.float32))
								
		return cv2.convertScaleAbs(prewitt)
		
	