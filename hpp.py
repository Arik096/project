import numpy as np
import cv2

im = cv2.imread('output2.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(im,contours,-1,(255,0,0),2)

for h,cnt in enumerate(contours):
    mask = np.zeros(im.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    mean = cv2.mean(im,mask = mask)

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 1)


cv2.imshow('image', im)
k = cv2.waitKey(0) & 0xFF
# press ESC to close  
if k == 27:  
    cv2.destroyAllWindows() 
# press s to save the output     
elif k == ord('s'):  
    cv2.imwrite('result_of_hpp_and_vpp.jpg',im) 
    cv2.destroyAllWindows() 