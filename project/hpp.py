import cv2
import numpy as np 
import matplotlib.pyplot as plt



img = cv2.imread("bangla.jpg")


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



img = cv2.bilateralFilter(img, 15, 75, 75) 




img = cv2.bitwise_not(img)
thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]



coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))




hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)
img = cv2.imread("bangla.jpg")
th = 2
H,W = img.shape[:2]
uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]


rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
for y in uppers:
    cv2.line(rotated, (0,y), (W, y), (255,0,0), 1)

for y in lowers:
    cv2.line(rotated, (0,y), (W, y), (0,255,0), 1)



plt.figure(figsize=(10,10))
plt.title('Original Image')
plt.imshow(rotated)
plt.show()