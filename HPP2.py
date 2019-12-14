import cv2
import matplotlib.pyplot as plt



img = cv2.imread("bangla.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10,10))



th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
plt.figure(figsize=(10,10))




pts = cv2.findNonZero(threshed)
ret = cv2.minAreaRect(pts)

(cx,cy), (w,h), ang = ret
if w>h:
    w,h = h,w
    ang += 90
ret


M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))
plt.figure(figsize=(10,10))


hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)


th = 2
H,W = img.shape[:2]
uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]


rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
for y in uppers:
    cv2.line(rotated, (0,y), (W, y), (255,0,0), 1)

for y in lowers:
    cv2.line(rotated, (0,y), (W, y), (0,255,0), 1)


cv2.imshow('image', rotated)


############ save output ##############
k = cv2.waitKey(0) & 0xFF
# press ESC to close  
if k == 27:  
    cv2.destroyAllWindows() 
# press s to save the output     
elif k == ord('s'):  
    cv2.imwrite('result.png',rotated) 
    cv2.destroyAllWindows() 
#########################################