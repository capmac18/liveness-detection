import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC

df1 = pd.read_csv("Real Face Data.csv")
df2 = pd.read_csv("Fake Face Data.csv")
df3 = pd.read_csv("Test Face Data.csv")

arr1 = np.array(df1)
arr2 = np.array(df2)
test_arr = np.array(df3)

l1 = len(arr1)
l2 = len(arr2)
l3 = len(test_arr)

# print(arr1[:5])
# print("#####")
# print(arr2[-5:])
# print("####")
X = np.concatenate((arr1,arr2),axis=0)


X = X[:,1:]
test_arr = test_arr[:,1:]
# print(X[:5])
# print("#####")
# print(X[-5:])
#X= preprocessing.scale(X)
#print(X[-5:,:])
# print(len(X),l1,l2)

m1, m2 = X[:, 0].min() , X[:, 0].max()
m3, m4 = X[:, 1].min(), X[:, 1].max()

m5, m6 = test_arr[:,0].min() , test_arr[:,0].max()
m7, m8 = test_arr[:,1].min() , test_arr[:,1].max()

for i in range(len(X)):
	X[i][0] = X[i][0]/(m2-m1)
	X[i][1] = X[i][1]/(m4-m3)

for i in range(len(test_arr)):
	test_arr[i][0] = test_arr[i][0]/(m2-m1)
	test_arr[i][1] = test_arr[i][1]/(m4-m3)

y = np.array([1]*l1 + [2]*l2)
# declaring the SVM classifier
#clf = make_pipeline(StandardScaler(), SVC(kernel='linear',gamma='auto'))
clf = svm.SVC(kernel='rbf',C=1000)

# fitting SVM on the dataset
clf.fit(X,y)

h = 0.1  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() , X[:, 0].max()
y_min, y_max = X[:, 1].min() , X[:, 1].max()

xx, yy = np.meshgrid(np.arange(x_min-0.1, x_max+0.1, (x_max-x_min)*0.01),
                     np.arange(y_min-0.1, y_max+0.1, (y_max-y_min)*0.01))
# print(xx)
# print("###")
# print(yy)

tmp = clf.decision_function(X)
#print(clf.decision_function([[0.02,0.03]]))
points_db = []

for i in range(len(xx)):
	for j in range(len(xx[0])):
		dist = clf.decision_function([[xx[i][j], yy[i][j]]])
		#print(dist[0])
		if abs(dist[0]) <= 0.3:
			points_db.append([xx[i][j], yy[i][j]])

points_db  = np.array(points_db)

print(clf.predict(test_arr))

print(clf.decision_function(test_arr))
plt.scatter(X[:, 0], X[:, 1], c = np.where(y==2,'tab:orange','tab:blue'))
plt.scatter([],[], c='tab:orange', label='Fake')
plt.scatter([],[], c='tab:blue', label='Real')
plt.scatter(points_db[:,0], points_db[:,1], c="#C34B73", marker='*',s=10,  label='decision_boundary')
plt.scatter(test_arr[:,0], test_arr[:,1], c='#432D28', label='test_ponts')
plt.xlabel('Edge Laplace Variance')
plt.ylabel('Edge Laplace Maximum')
plt.legend()
plt.show()


# fig, ax = plt.subplots()
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# ax.contourf(xx, yy, Z)
# ax.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
# plt.scatter(test_arr[:,0], test_arr[:,1], c='g')
# plt.show()

# w = clf.coef_[0]
# a = -w[0] / w[1]
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# xx = np.arange(x_min, x_max, (x_max-x_min)*0.1)
# yy = a * xx - clf.intercept_[0] / w[1]
# h0 = plt.plot(xx, yy, 'k-', label="non weighted div")
# plt.scatter(X[:, 0], X[:, 1], c = y)
# plt.show()

# print(xx)

# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, m_max]x[y_min, y_max].
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)


# # print(arr1[:5,:],"\\n",arr2[:5,:])

# plt.scatter([i[1] for i in arr1],[i[2] for i in arr1], label='Real')
# plt.scatter([i[1] for i in arr2],[i[2] for i in arr2], label='Fake')
# plt.legend(loc="lower right")
# plt.xlabel("Edge-Laplace   Variance")
# plt.ylabel("Edge-Laplace   Maximum")
# plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
# plt.show()