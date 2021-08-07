from PIL import Image, ImageFilter, ImageOps
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from skimage.feature import local_binary_pattern as lbp


# def process(arr):
# 	h = arr.shape[0]
# 	w = arr.shape[1]
# 	visited = np.array([[-1 for _ in range(w)] for _ in range(h)])
# 	q = []
# 	q.append((int(h/2),int(w/2),1))

# 	record = [0]
# 	visited[int(h/2)][int(w/2)] = 0
# 	size = [0]
# 	while(len(q)):
# 		i,j,c = q.pop(0)
# 		#print(i,j,c)
# 		if c==len(record):
# 			record[-1] += arr[i][j]
# 			size[-1] += 1
# 		else:
# 			record.append(arr[i][j])
# 			size.append(1)

# 		factors = [0,1,0,-1,1,0,-1,0,1,-1,1,1,-1,1,-1,-1]
# 		for f in range(0,16,2):
# 			ni = i + factors[f]
# 			nj = j + factors[f+1]
# 			if ni<0 or nj<0 or ni>=h or nj>=w or visited[ni][nj]!=-1:
# 				continue
# 			q.append((ni,nj,c+1))
# 			visited[ni][nj] = c

# 	for i in range(h):
# 		for j in range(w):
# 			visited[i][j] = record[visited[i][j]]

# 	print(len(record), len(size))
# 	for i in range(len(record)):
# 		record[i] = int(record[i]/size[i])

# 	return record,visited


# img1 = Image.open("compare/real15.jpeg")
# img2 = Image.open("compare/fake15.jpeg")

# s1 = img1.filename
# s2 = img2.filename

# img1 = img1.resize((601,801))
# img2 = img2.resize((601,801))

# img1_GS = ImageOps.grayscale(img1)
# img2_GS = ImageOps.grayscale(img2)

# img1_GS.show()
# img2_GS.show()

# avg1, nimg1 = process(np.array(img1_GS))
# avg2, nimg2 = process(np.array(img2_GS))

# nimg1 = Image.fromarray(nimg1)
# nimg2 = Image.fromarray(nimg2)

# nimg1.show()
# nimg2.show()

# plt.scatter([i for i in range(len(avg1))], avg1, label = 'Real', color = 'r')
# plt.scatter([i for i in range(len(avg2))], avg2, label = 'Fake')
# plt.title(s1+"\n"+s2)
# plt.legend()
# plt.show()

def process2(arr1, arr2):
	h = arr1.shape[0]
	w = arr2.shape[1]

	avg = np.zeros(256)
	freq = np.zeros(256)

	for i in range(h):
		for j in range(w):
			avg[arr1[i][j]] += arr2[i][j]
			freq[arr1[i][j]] += 1

	for i in range(256):
		avg[i] = (avg[i]/freq[i])

	return avg


img1 = Image.open("compare/187.jpg")
img2 = Image.open("compare/188.jpg")
img3 = Image.open("compare/29.jpeg")
img4 = Image.open("compare/30.jpeg")

s1 = img1.filename
s3 = img3.filename

img1 = img1.resize((601,801))
img2 = img2.resize((601,801))
img3 = img3.resize((601,801))
img4 = img4.resize((601,801))

img1 = ImageOps.grayscale(img1)
img2 = ImageOps.grayscale(img2)
img3 = ImageOps.grayscale(img3)
img4 = ImageOps.grayscale(img4)

img1.show()
img2.show()
img3.show()
img4.show()

avg1 = process2(np.array(img1), np.array(img2))
avg2 = process2(np.array(img3), np.array(img4))

plt.scatter([i for i in range(len(avg1))], avg1, label = 'Real', color = 'r')
plt.scatter([i for i in range(len(avg2))], avg2, label = 'Fake')
plt.title(s1+"\n"+s3)
plt.legend()
plt.show()