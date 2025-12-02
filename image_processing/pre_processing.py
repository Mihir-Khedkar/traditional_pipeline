import cv2
import numpy as np

class PreProcessor:
	def __init__(self, img):
		# print("Initialising Image..........")
		# print(type(image))
		self.image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
	def gaussian_blur(self, kernel_size=(5,5), sigma=0):
		return cv2.GaussianBlur(self.image, kernel_size, sigma)
		
	def bilateral_denoise(self, diameter=9, sigma_color=75, sigma_space=75):
		return cv2.bilateralFilter(self.image, diameter, sigma_color, sigma_space)
	
	def histogram_equalization(self):
		return cv2.equalizeHist(self.image)
		
	def clahe(self, clip_limit=2.0, title_grid_size=(8,8)):
		clahe = cv2.createCLAHE(clipLimit=clip_limit, titleGridSize=title_grid_size)
		return clahe.apply(self.image)