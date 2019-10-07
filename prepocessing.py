import numpy as np 
import cv2 
  
########### taking image input ############
img = cv2.imread('input.jpg', 255) 
###########################################

############# apply greyscal filter ########
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
############################################

######## apply bilateral filter ############
img = cv2.bilateralFilter(img, 15, 75, 75) 
############################################

####### image binariazation ################
img = cv2.bitwise_not(img)
thresh = cv2.threshold(img, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
coords = np.column_stack(np.where(thresh > 0))
############################################

########### skew angle correction ###########
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
################################################################

########## showing output  ##########
cv2.imshow('image', rotated)
#####################################

############ save output ##############
k = cv2.waitKey(0) & 0xFF
# press ESC to close  
if k == 27:  
    cv2.destroyAllWindows() 
# press s to save the output     
elif k == ord('s'):  
    cv2.imwrite('output.jpg',rotated) 
    cv2.destroyAllWindows() 
#########################################