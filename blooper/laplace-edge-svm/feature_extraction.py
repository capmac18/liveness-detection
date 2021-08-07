import numpy as np 
from scipy.ndimage import variance
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace, sobel
from skimage.transform import resize
import pandas as pd
import os

ls = []
os.chdir('.\\Data\\All_Fake')
for f in os.listdir('.'):
	if f.endswith('.jpg') or f.endswith('.jpeg'):
		path = f
		# path = f'{os.getcwd()}\\Data{f}'
		#print(path)
		img = io.imread(path)

		# # Resize image
		img = resize(img, (600, 800))

		# Grayscale image
		img = rgb2gray(img)

		# Edge detection
		edge_laplace = laplace(img, ksize=3)
		var = variance(edge_laplace)
		m2 = np.amax(edge_laplace)
		ls.append([var, m2])


		# Print output
		# print(f"Variance: {variance(edge_laplace)}")
		# print(f"Maximum : {np.amax(edge_laplace)}")
		# print(f"Mean: {np.mean(edge_laplace)}")

df = pd.DataFrame(ls)
df.to_csv("Fake Face Data.csv")