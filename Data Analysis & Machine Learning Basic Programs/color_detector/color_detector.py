import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("no/colours.jpg")
# plt.figure(figsize=(20,8))
# plt.imshow(img)
# plt.show()

grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(20,8))
# plt.imshow(grid_RGB)
# plt.show()


grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)
# FOR YELLOW
lower = np.array([25,150,50])  #Yellow LOWER HSV
upper = np.array([35,255,255]) #Yellow UPPER HSV

mask = cv2.inRange(grid_HSV,lower,upper)
# plt.figure(figsize=(20,8))
# plt.imshow(mask)
# plt.show()

res = cv2.bitwise_and(grid_RGB,grid_RGB,mask=mask)
plt.figure(figsize=(20,8))
plt.imshow(res)
plt.show()


# FOR RED
lower1 = np.array([0,150,50])  #Yellow LOWER HSV
upper1 = np.array([10,255,255]) #Yellow UPPER HSV

mask1 = cv2.inRange(grid_HSV,lower1,upper1)

res1 = cv2.bitwise_and(grid_RGB,grid_RGB,mask=mask1)
plt.figure(figsize=(20,8))
plt.imshow(res1)
plt.show()

# FOR DARK BLUE
lower2 = np.array([115,150,50])  #Yellow LOWER HSV
upper2 = np.array([125,255,255]) #Yellow UPPER HSV

mask2 = cv2.inRange(grid_HSV,lower2,upper2)

res2 = cv2.bitwise_and(grid_RGB,grid_RGB,mask=mask2)
plt.figure(figsize=(20,8))
plt.imshow(res2)
plt.show()