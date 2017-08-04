# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:01:12 2017

@author: petersenadmin
"""

import matplotlib.pyplot as plt
import scipy.misc
img = plt.imread('image8.jpg')
img_copy = img.copy()
lx, ly = img_copy.shape[0], img_copy.shape[1]
#img_crop = img_copy[int(lx / 6): int(- lx / 6), int(ly / 5): int(- ly / 5)]
img_resized = scipy.misc.imresize(img_copy, (32, 32))
plt.imshow(img_resized)
plt.imsave('img8.jpg', img_resized)