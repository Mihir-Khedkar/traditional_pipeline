import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

class Histogram:
	
	def __init__(self, title, xlabel, ylabel, label="None"):
		self.title = title
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.label = label
		print("Image ready for Histogram creation")
	
	def create_histogram(self, img_list, labels, output_path = r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\outputs', xlbl = "Pixel Intensity", ylbl  = "Frequency", lower_limit=0, high_limit=256):
		
		plt.figure(figsize=(8,4))
		plt.title(self.title)
		plt.xlabel(xlbl)
		plt.ylabel(ylbl)
		
		i=0
		colors = ['black', 'green', 'red', 'orange', 'cyan']
		
		for img in img_list:
			tmp_hist = cv2.calcHist([img], [0], None, [32], [lower_limit,high_limit])
			bin_edges = np.linspace(lower_limit, high_limit, len(tmp_hist)+1)
			bin_centers  = (bin_edges[:-1] + bin_edges[1:])
			plt.plot(bin_centers, tmp_hist, color=colors[i], label=labels[i])
			plt.legend()
			i=+1
		
		plt.xlim([lower_limit, high_limit])
		plt.grid(True , linestyle='--', alpha=0.5)
		
		file_name = self.title + ".png"
		file_path = os.path.join(output_path, file_name)
		plt.savefig(file_path, bbox_inches='tight')
		
		print(f"Successfully created histogram {file_name}")

	def plot2histograms(self, list1, file_name, output_path, lower_end=0, higher_end=500, bins=30):
		plt.figure(figsize=(8,5))
		plt.hist(list1, bins=bins, alpha=0.5, label=self.label)
		plt.xlim(lower_end, higher_end)
		plt.xlabel(self.xlabel)
		plt.ylabel(self.ylabel)
		plt.title(self.title)
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(output_path, file_name), dpi=300)
		print(f"Successfully generated Histogram {file_name} at {output_path}")
