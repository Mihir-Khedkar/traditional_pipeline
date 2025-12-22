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

	def create_2d_patch_histogram(self, img, patch_size=(16,16), count_mode='nonzero', threshold=None,
					output_path=r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\outputs',
					file_name=None, cmap='viridis', show=False):
		
		if img is None:
			raise ValueError("img must be a numpy array")

		# Convert to grayscale if needed
		if len(img.shape) == 3 and img.shape[2] == 3:
			img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		else:
			img_gray = img.copy()

		# Optionally threshold to binary image
		if threshold is not None:
			_, img_proc = cv2.threshold(img_gray, int(threshold), 255, cv2.THRESH_BINARY)
		else:
			img_proc = img_gray

		ph, pw = patch_size
		h, w = img_proc.shape[:2]

		# Pad image so dimensions are divisible by patch size
		n_rows = (h + ph - 1) // ph
		n_cols = (w + pw - 1) // pw
		pad_h = n_rows * ph - h
		pad_w = n_cols * pw - w
		if pad_h > 0 or pad_w > 0:
			img_proc = np.pad(img_proc, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

		counts = np.zeros((n_rows, n_cols), dtype=float)

		for i in range(n_rows):
			for j in range(n_cols):
				sy = i * ph
				x = j * pw
				patch = img_proc[sy:sy+ph, x:x+pw]
				if count_mode == 'nonzero':
					counts[i, j] = np.count_nonzero(patch)
				elif count_mode == 'sum':
					counts[i, j] = float(np.sum(patch))
				elif count_mode == 'mean':
					counts[i, j] = float(np.mean(patch))
				else:
					raise ValueError("count_mode must be 'nonzero', 'sum' or 'mean'")

		# Visualization
		plt.figure(figsize=(6,5))
		plt.imshow(counts, cmap=cmap, interpolation='nearest', origin='upper')
		plt.colorbar(label='Count')
		plt.title(self.title + ' - 2D Patch Histogram')
		plt.xlabel(f'Patch X (width={pw}px)')
		plt.ylabel(f'Patch Y (height={ph}px)')

		if file_name is None:
			file_name = f"{self.title}_2d_patch_histogram.png"

		os.makedirs(output_path, exist_ok=True)
		file_path = os.path.join(output_path, file_name)
		plt.tight_layout()
		plt.savefig(file_path, dpi=300)

		if show:
			plt.show()
		else:
			plt.close()

		print(f"Saved 2D patch histogram to {file_path}")

		return counts
