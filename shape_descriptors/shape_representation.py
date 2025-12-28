import numpy as np
import cv2

class ContourAnalysis:
	def __init__(self, contour):
		self.contour = contour
		
	def compute_features(self, contour):
		features = {}
		area = cv2.contourArea(contour)
		perimeter = cv2.arcLength(contour, True)
		features["area"] = area
		features["perimeter"] = perimeter
		
		x, y, w, h = cv2.boundingRect(contour)
		features["bbox"] = (x, y, w, h)
		features["aspect_ratio"] = w / h if h > 0 else 0
		
		
		hull = cv2.convexHull(contour)
		hull_area  = cv2.contourArea(hull)
		features["extent"] = area / (hull_area + 1e-5)
		features["solidity"] = area / (hull_area + 1e-5)
		
		moments = cv2.moments(contour)
		features["hu_moments"] = cv2.HuMoments(moments).flatten()
		
		return features
	
	def classify(self, features):
		ar = features["aspect_ratio"]
		solidity = features["solidity"]
		extent = features["extent"]
		area = features["area"]
		
		if area < 100:
			return "noise"
			
		if ar > 4:
			return "stiffner"
			
		if 0.75 < solidity < 1.0 and extent > 0.6:
			return "plate"
			
		if solidity < 0.7:
			return "cutout_or_hole"
			
		return "unknown"
		
	def visualize(self, image):
		out = image.copy()
		
		for cnt in self.contour:
			feats = self.compute_features(cnt)
			label = self.classify(feats)
			
			x, y, w, h = feats["bbox"]
			cx, cy = x + w //2, y + h//2
			
			cv2.drawContours(out, [cnt], -1, (0,255,0), 2)
			cv2.putText(out, label, (cx, cy),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
				
		return out
	
class BoundingBoxInfo:
	def __init__(self, bb_annot: list, oriented_bb_annot: list) -> None:
		self.bb_annot = bb_annot
		self.oriented_bb_annot = oriented_bb_annot

	def bbarea(self) -> list:
		bb_areas = []
		for elements in self.bb_annot:
			x,y,w,h = elements
			area = w*h
			bb_areas.append(area)
		return bb_areas
	
	def oribbarea(self) -> list:
		oriented_bb_areas = []
		for elements in self.oriented_bb_annot:
			(cx,cy),(w,h),angle = elements
			width = int(w)
			height = int(h)
			area = width * height
			oriented_bb_areas.append(area)
		return oriented_bb_areas

	def draw(self, image, fill=False):
		for elements in self.bb_annot:
			x,y,w,h = elements			
			if fill == True:
				cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), thickness=cv2.filled)
			else:
				cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
		
		for elements in self.oriented_bb_annot:			
			box = cv2.boxPoints(elements)
			box = np.int0(box)
			if fill == True:
				cv2.drawContours(image, [box], 0, (255,0,0), thickness=cv2.filled)
			else:
				cv2.drawContours(image, [box], 0, (0,0,255), 2)

		return image