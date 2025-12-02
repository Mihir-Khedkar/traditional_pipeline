import cv2
import numpy as np
from typing import List, Tuple, Dict

class Morphological:
	
	def __init__(self, kernel: np.array = None, kernel_size: int=3, shape= cv2.MORPH_RECT) -> None:
		self.kernel_size = kernel_size
		self.shape = shape
		self.kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
		
	def erode(self, image: np.ndarray, iterations: int=1) -> np.ndarray:
		return cv2.erode(image, self.kernel, iterations=iterations)
	
	def dilate(self, image:np.ndarray, iterations: int=1) -> np.ndarray:
		return cv2.dilate(image, self.kernel, iterations=iterations)
		
	def close(self, image: np.ndarray, iterations: int=1) -> np.ndarray:
		return cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.kernel, iterations=iterations)
		
	def open(self, image: np.ndarray, iterations: int=1) ->  np.ndarray:
		return cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel, iterations=iterations)
		
	def gradient(self, image: np.ndarray, iterations: int=1) -> np.ndarray:
		return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, self.kernel)
		
class ContourExtraction:
	
	def __init__(self, retrieval_mode: int=cv2.RETR_TREE, approximation: int=cv2.CHAIN_APPROX_SIMPLE) -> None: 
		self.retrieval_mode = retrieval_mode
		self.approximation = approximation
		
	def preprocess(self, image: np.ndarray) -> np.ndarray:
		if len(image.shape) == 3:
			gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		else:
			gray_image = image.copy()
			
		_, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		
		return binary
		
	def extract(self, image: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
		contours, hierarchy = cv2.findContours(image, self.retrieval_mode, self.approximation)
		print(f"Extracted contours from the Given Image {len(contours)}")
		return contours, hierarchy
		
	def draw(self, image: np.ndarray, contours: List[np.ndarray], thickness: int=2) -> np.ndarray:
		canvas = image.copy()
		cv2.drawContours(canvas, contours, contourIdx=-1, color=(0, 0, 255), thickness=thickness)
		print("Drawn the contours on the canvas")
		return canvas
		
	def fill(self, image: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
		canvas = cv2.fillPoly(image, contours, (0,0,0))
		print("Filled the contours on the canvas")
		return canvas