'impot libraries and packeges'
import cv2 
import numpy as np
from matplotlib import pyplot as plt

IMG_DIR = 'images/'


'taking input'
image = cv2.imread(IMG_DIR + 'img1.jpg')

plt.figure(figsize=(7,7))
plt.title('Original Image')
plt.imshow(image)
plt.show()


"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~ image pre-processing ~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""


'applying greyscale filter'
gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(7,7))
plt.title('GrayScale Image')
plt.imshow(gray_scale_image, cmap='gray')
plt.show()


'applying bilateral filter'
bilateral_image = cv2.bilateralFilter(gray_scale_image, 15, 75, 75)

plt.figure(figsize=(7,7))
plt.title('Bilateral Image')
plt.imshow(gray_scale_image, cmap='gray')
plt.show()


'applying bit plane slicing or getting binary image'
binary_image = cv2.bitwise_not(bilateral_image)

plt.figure(figsize=(7,7))
plt.title('Binary Image')
plt.imshow(binary_image)
plt.show()


'applying skew angle correction'

thresh = cv2.threshold(binary_image, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle
(h, w) = binary_image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_image = cv2.warpAffine(binary_image, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.putText(rotated_image, "Angle: {:.2f} degrees".format(angle),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

plt.figure(figsize=(7,7))
plt.title('Rotated Image')
plt.imshow(rotated_image)
plt.show()







