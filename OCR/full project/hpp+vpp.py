import cv2
import numpy as np
IMG_DIR = 'images/'


'taking input'
img = cv2.imread(IMG_DIR + 'img1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

"Finding threshold"
th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

"Finding rectangle or bounding box"
pts = cv2.findNonZero(threshed)
ret = cv2.minAreaRect(pts)

(cx,cy), (w,h), ang = ret


"Rotating image for alignment of box"
M = cv2.getRotationMatrix2D((cx,cy), 0, 1.0)
rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))
output = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))

def lines_seperator():
    hist2 = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)
    upper = 0
    lower = len(hist2)-1
    while hist2[upper] <= min(hist2):
        upper+=1
    while hist2[lower] <= min(hist2):
        lower-=1
    x = upper
    prev_x = upper
    while x < lower:
        while x < len(hist2) and hist2[x] <= min(hist2):
            x += 1
        prev_x = x
        while x < len(hist2) and hist2[x] > min(hist2):
            x += 1
        #print(prev_x,x)
        if x - prev_x > 10:
            cv2.line(output, (0, x),(WIDTH, x), (0,0,255),1)
            cv2.line(output, (0, prev_x),(WIDTH, prev_x), (0,255,0),1)
            words_seperator(x, prev_x)


def words_seperator(lower, upper):
    hist2 = cv2.reduce(rotated[upper:lower],0, cv2.REDUCE_AVG).reshape(-1)
    left = 0
    right = len(hist2)-1
    while hist2[left] <= min(hist2):
        left+=1
    while hist2[right] <= min(hist2):
        right-=1
    x = left
    prev_x = left

    while x < right:
        while x < len(hist2) and hist2[x] <= min(hist2):
            x += 1
        prev_x = x
        while x < len(hist2) and hist2[x] > min(hist2):
            x += 1
        #print(prev_x,x)    
        if x - prev_x >5:
            cv2.line(output, (prev_x,upper),(x, upper), (0,0,255),1)
            cv2.line(output, (prev_x,lower),(x, lower), (0,0,255),1)
            cv2.line(output, (prev_x,lower),(prev_x, upper), (0,0,255),1)
            cv2.line(output, (x,lower),(x, upper), (0,0,255),1)
           



           

"Drawing lines or seperation of lines"
hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)

HEIGHT, WIDTH = img.shape[:2]

output2 =cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
output = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

lines_seperator()

import matplotlib.pyplot as plt



cv2.imwrite("result.png", output)
