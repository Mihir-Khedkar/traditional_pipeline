import cv2
import numpy as np
import json
import os

class ImageRaw:
	def __init__(self, image_name, input_path):
		self.image_path = input_path
		self.image_name = image_name
		self.image = None
		self.attributes = {}
		
	def load_image(self):
		if os.path.exists(self.image_path):
			print(f"Loaded image: {self.image_name}")
		else:
			raise FileNotFoundError(f"Image {self.image_name} not found at: {self.image_path}")
			
		complete_path = os.path.join(self.image_path, self.image_name)
		print("Image path is:" + complete_path)
		self.image = cv2.imread(complete_path)
		
		if self.image is None:
			raise ValueError("Failed to load image. Check provided image path and name.")
		
		height, width, channels = self.image.shape
		self.attributes = {
			"filename" : complete_path, 
			"width" : width,
			"height" : height,
			"channels" : channels,
			"dtype" : str(self.image.dtype),
			"size" : os.path.getsize(complete_path)
		}
		
		print(f"Loaded {self.image_name} with height: {height} and width: {width} and channels: {channels}")
		
	def show_image(self, window_name="RawImage"):
		if self.image is None:
			raise ValueError("Cannot show image. Call load_image first")
			
		cv2.imshow(window_name, self.image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
class ImageProcessed(ImageRaw):
	def __init__(self, image):
		self.image = image
		self.attributes = {}
	
	def show_attributes(self):
		height, width, channels = self.image.shape
		self.attributes = {
			"height" : height,
			"width" : width,
			"channels" : channels,
			"dtype" : self.image.dtype
			}
		
		print(f"Image attributes are: H {height}, W {width}, C {channels}, dtype {self.attributes['dtype']}")