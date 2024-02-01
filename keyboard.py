import cv2
import numpy as np

# Read the input image
image = cv2.imread('Piano_Keyboard.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection to the grayscale image
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply Hough line transform to detect lines in the image
#lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
    
if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
    
# Display the image with the detected lines and valid combinations
cv2.imshow('Keyboard Location', image)
cv2.imshow('Keyboard gray', edges)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
cv2.waitKey(0)
cv2.destroyAllWindows()