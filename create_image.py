
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def normalize(x, d):
	return int(x * d)

def scaleUp(mat):
	vecfunc = np.vectorize(normalize)
	res = vecfunc(mat,255)
	res = res.reshape((10000,32,16))
	return res

viewLeft = np.load('test-view1_L.npy')
viewRight = np.load('test-view1_R.npy')

orig = np.load('../../cifar10/matpic/test/view1.npy')
orig1 = np.load('../../cifar10/matpic/test/view2.npy')
print orig.shape
orig = orig.reshape((10000,32,16))
orig1 = orig1.reshape((10000,32,16))
pic1 = np.dstack((orig,orig1))
print ("pic1", pic1.shape)
print ("orig", orig.shape)

Left = scaleUp(viewLeft)
Right = scaleUp(viewRight)

print ("Left", Left.shape)
print ("Right", Right.shape)

pic2 = np.dstack((Left, Right))


plt.imshow(pic1[3000], cmap = plt.get_cmap('gray'))
plt.show()
plt.imshow(pic2[3000], cmap = plt.get_cmap('gray'))
plt.show()