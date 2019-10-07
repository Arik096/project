import cv2
import numpy as np

im = cv2.imread('input_2.jpg', cv2.IMREAD_GRAYSCALE)


im = 255 - im


proj = np.sum(im,1)


m = np.max(proj)
w = 500
result = np.zeros((proj.shape[0],500))


for row in range(im.shape[0]):
   cv2.line(result, (0,row), (int(proj[row]*w/m),row), (255,255,255), 1)


cv2.imwrite('output_2.jpg', result)