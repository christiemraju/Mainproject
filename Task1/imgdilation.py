import cv2
import numpy as np

img = np.array(cv2.imread('input2.png',0))
b=np.array([1,1,0,1,0,1,0,1])
d = np.empty_like(img)
a=np.pad(img,pad_width=1,mode='constant',constant_values=0)

a = np.delete(a, 0, 0)
a = np.delete(a, len(a)-1, 0)

for i in range(len(a)-(int(b.shape[0]/2))):
    for j in range(1,len(a[i])-1-(int(b.shape[0]/2))):
        count = 0
        for k in a[i][j-int(b.shape[0]/2):j+1+int(b.shape[0]/2)]:
            for l in b:
                count= count + k*l
        if count == 0:
            d[i][j]=0
        else:
            d[i][j]=255
cv2.imshow('Input', img)
cv2.imshow('Dilation', d)

cv2.waitKey(0)
