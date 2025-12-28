import cv2
import numpy as np

class EdgeDetector:
	def __init__(self, image):
		self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
	def canny_edge(self, lower_threshold=100, upper_threshold=200):
		edges = cv2.Canny(self.image, lower_threshold, upper_threshold, L2gradient=True)
		return edges
	
	def sobel(self, dx=1, dy=1, ksize=3):
		sobel = cv2.Sobel(self.image, cv2.CV_64F, dx, dy, ksize=ksize)
		return cv2.convertScaleAbs(sobel)
		
	def scharr(self, dx=1, dy=1):
		scharr = cv2.Scharr(self.image, cv2.CV_64F, dx, dy)
		return cv2.convertScaleAbs(scharr)
		
	def prewitt(self):
	
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

	def canny_scharr_gradient(self):
		canny_gx = cv2.Scharr(self.image, cv2.CV_64F, dx=1, dy=0)
		canny_gy = cv2.Scharr(self.image, cv2.CV_64F, dx=0, dy=1)
		mag, ang = cv2.cartToPolar(canny_gx, canny_gy, angleInDegrees=True)
		self.mag = mag
		self.ang = ang

	def canny_non_max_suppression(self, wk_multiplier=0.1, str_multiplier=0.5):
		mag = self.mag
		max_mag = np.max(mag)
		
		self.weak_thresh = wk_multiplier*max_mag
		self.strong_thresh = str_multiplier*max_mag

		height, width = self.image.shape
# Angles: 0, 22.5 , 45, 67.5, 90, 112.5, 135, 157.5, 180
		for i_x in range(width):
			for i_y in range(height):
				grad_ang = self.ang[i_y, i_x] 
				if abs(grad_ang) > 180:
					grad_ang = abs(grad_ang - 180)
				else: 
					abs(grad_ang)

				if grad_ang <= 22.5:
					n1_x, n1_y = i_x-1, i_y
					n2_x, n2_y = i_x+1, i_y

				elif grad_ang > 22.5 and grad_ang <= 67.5:
					n1_x, n1_y = i_x-1, i_y+1
					n2_x, n2_y = i_x+1, i_y-1

				elif grad_ang > 67.5 and grad_ang <= 112.5:
					n1_x, n1_y = i_x, i_y+1
					n2_x, n2_y = i_x, i_y-1

				elif grad_ang > 112.5 and grad_ang <= 157.5:
					n1_x, n1_y = i_x, i_y+1
					n2_x, n2_y = i_x, i_y-1

				else:
					n1_x, n1_y = i_x-1, i_y
					n2_x, n2_y = i_x+1, i_y

				if 0 <= n1_x < width and 0 <= n1_y < height:
					if mag[i_y,i_x] < mag[n1_y, n1_x]:
						mag[i_y, i_x] = 0
						continue
				
				if 0 <= n2_x < width and 0 <= n2_y < height:
					if mag[i_y,i_x] < mag[n2_y, n2_x]:
						mag[i_y, i_x] = 0
						continue

		self.new_mag = mag

	def canny_hysteresis(self):
		mag = self.new_mag
		ids = np.zeros_like(self.image)
		height, width = ids.shape
		for i_x in range(width):
			for i_y in range(height):
				grad_mag = mag[i_y, i_x]

				if grad_mag < self.weak_thresh:
					ids[i_y, i_x] = 0
				elif self.weak_thresh <= grad_mag < self.strong_thresh:
					ids[i_y, i_x] = 0.5
				else:
					ids[i_y, i_x] = 1

		edges = ids*255
	
		return edges.astype(int)