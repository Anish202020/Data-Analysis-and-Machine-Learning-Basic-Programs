import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("mountain/mountain.jpeg")

grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

# DETECT THE GREEN GLASS

lower = np.array([35,150,50])
upper = np.array([75,255,255])

mask = cv2.inRange(grid_HSV,lower,upper)

res = cv2.bitwise_and(grid_RGB,grid_RGB,mask=mask)
plt.figure(figsize=(20,8))
plt.imshow(res)
plt.show()