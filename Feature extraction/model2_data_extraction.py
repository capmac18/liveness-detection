import numpy as np 
import pandas as pd
import os
from PIL import Image, ImageFilter, ImageOps

def process(arr1, arr2):
	h = arr1.shape[0]
	w = arr2.shape[1]

	avg = np.zeros(256)
	freq = np.zeros(256)

	for i in range(h):
		for j in range(w):
			avg[arr1[i][j]] += arr2[i][j]
			freq[arr1[i][j]] += 1

	for i in range(256):
		if freq[i]==0:
			avg[i] = np.NaN
		else:
			avg[i] = (avg[i]/freq[i])

	return avg

os.chdir('.\\Data\\Low exposure model\\Test\\Real')

names = [i for i in os.listdir('.')]
names.sort()
ls = []

limit = len(names)

for i in range(1,limit,2):
	f1 = str(i)
	f2 = str(i+1)
	if ((f1+'.jpg') in names):
		f1 = f1+'.jpg'
	else:
		f1 = f1+'.jpeg'
	if ((f2+'.jpg') in names):
		f2 = f2+'.jpg'
	else:
		f2 = f2+'.jpeg'

	print(f1, f2)
	img1 = Image.open(f1)
	img2 = Image.open(f2)

	img1 = img1.resize((601,801))
	img2 = img2.resize((601,801))

	img1 = ImageOps.grayscale(img1)
	img2 = ImageOps.grayscale(img2)

	avg = process(np.array(img1), np.array(img2))
	ls.append(avg)

df = pd.DataFrame(ls)

for col in df.columns:
	# if df[col].mean()==np.NaN:
	# 	print("ok")
	# 	df[col] = pd.DataFrame(np.zeros(len(df[col])))
	# else:
	df[col] = df[col].replace(np.NaN, df[col].mean())

#df = df.fillna(0)
print(df.head())
df.to_csv("TestReal_model2.csv")

