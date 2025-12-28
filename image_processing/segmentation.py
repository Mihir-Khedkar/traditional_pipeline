from email.mime import image
import cv2 
import numpy as np 

class Segmentation:
	def __init__(self, image):
		self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		print("Received Image for Segmentation")
		
	def threshold_global(self, T):
		_, binary = cv2.threshold(self.image, T, 255, cv2.THRESH_BINARY)
		return binary
	
	def threshold_otsu(self):
		_, binary = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		return binary
		
	def threshold_adaptive_mean(self, block_size=11, c=2):
		binary = cv2.adaptiveThreshold(
			self.image, 255,
			cv2.ADAPTIVE_THRESH_MEAN_C,
			cv2.THRESH_BINARY,
			block_size, callable
		)
		return binary
		
	def thresh_adaptive_gaussian(self, block_size=11, c=2):
		binary = cv2.adaptiveThreshold(
			self.image, 255,
			cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
			cv2.THRESH_BINARY,
			block_size, c
		)
		return binary
	
	def region_growing(self, seed_point, threshold=5):
		x,y = seed_point
		seed_value = self.gray[y,x]
		
		visited = np.zeros_like(self.image, dtype=bool)
		mask = np.zeroes_like(self.image, dtype=np.uint8)
			
		stack = [(x,y)]
		
		while stack:
			cx, cy = stack.pop()
			if visited[cy, cx]:
				continue
			
			visited[cy,cx] = True
			if abs(int(self.image[cy,cx]) - int(seed_value)) < threshold:
				mask[cy,cx] = 255
				
				for nx, ny in [(cx-1,cy), (cx+1,cy), (cx,cy-1), (cx,cy+1)]:
					if 0 <= nx < self.image.shape[1] and 0 <= ny < self.image.shape[0]:
						if not visited[nx,ny]:
							stack.append((nx,ny))
		
		return mask
		
	def watershed_segmentation(self, image, thresh):
		kernel = np.ones((3,3), np.uint8)
		opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
		
		sure_bg = cv2.dilate(opening, kernel, iterations=3)
		
		dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
		_, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
		
		sure_fg = np.uint8(sure_fg)
		unknown = cv2.subtract(sure_bg, sure_fg)
		
		num_labels, markers = cv2.connectedComponents(sure_fg)
		markers = markers + 1
		markers[unknown == 255] = 0
		
		# result = image.copy()
		result = np.copy(image)
		cv2.watershed(result, markers)
		# cv2.watershed(result, markers)
		# cv2.watershed(result, markers)
		result[markers == -1] = [255,0,255] 
		
		return result