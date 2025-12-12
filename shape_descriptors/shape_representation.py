import numpy as np
import cv2

class ContourAnalysis:
	def __init__(self, contour, image):
		self.image = image
		self.contour = contour
		
	def compute_features(self, contour):
		features = {}
		area = cv2.contourArea(contour)
		perimeter = cv2.arcLenght(contour, True)
		features["area"] = area
		features["perimeter"] = perimeter
		
		x, y, w, h = cv2.boundingRect(contour)
		features["bbox"] = (x, y, w, h)
		features["aspect_ratio"] = w / h if h > 0 else 0
		
		features["extent"] = area / (hull_area + 1e-5)
		
		hull = cv2.convexHull(contour)
		hull_area  = cv2.contourArea(hull)
		features["solidity"] = area / (hull_area + 1e-5)
		
		moments = cv2.moments(contour)
		features["hu_moments"] = cv2.HuMoments(moments).flatten()
		
		return features
	
	def classify(self, features):
		ar = features["aspect_ratio"]
		solidity = features["solidity"]
		extent = features["extent"]
		area = features["area"]
		
		if area < 200:
			return "noise"
			
		if ar > 4:
			return "stiffner"
			
		if 0.75 < solidity < 1.0 and extent > 0.6:
			return "plate"
			
		if solidity < 0.7:
			return "cutout_or_hole"
			
		return "unknown"
		
	def visualize(self):
		out = self.image.copy()
		
		for cnt in self.contours:
			feats = self.compute_features(cnt)
			label = self.classify(feats)
			
			x, y, w, h = feats["bbox"]
			cx, cy = x + w //2, y + h//2
			
			cv2.drawContours(out, [cnt], -1, (0,255,0), 2)
			cv2.putText(out, label, (cx, cy),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
				
		return out