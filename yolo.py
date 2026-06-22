import cv2
import numpy as np
img=cv2.imread(r"C:\Users\Johaan Liju\downloads\rectangular.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50,150)
cv2.imshow("edges",edges)
cv2.waitKey(0)
contours, hierarchy= cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
big = max(contours,key=cv2.contourArea)
print(big.shape)
epsilon = 0.02 * cv2.arcLength(big, True)
approx = cv2.approxPolyDP(big,epsilon,True)
print(approx.shape)
corners = approx.reshape(4,2)
sums = corners[:,0] + corners[:,1]
diffs = corners[:, 0] - corners[:, 1]

top_left = corners[np.argmin(sums)]
bottom_right = corners[np.argmax(sums)]
top_right = corners[np.argmin(diffs)]
bottom_left = corners[np.argmax(diffs)]
ordered=np.array([top_left,top_right,bottom_left,bottom_right])
print(ordered)
for i, point in enumerate(ordered):
    x, y = point
    x, y = int(x), int(y)  
    cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
    cv2.putText(img, str(i), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imshow("corners", img)
cv2.waitKey(0)
