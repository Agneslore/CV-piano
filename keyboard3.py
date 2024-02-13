import cv2
import numpy as np
import statistics


# input
image = cv2.imread('above.jpeg')
ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# filter thresholds
lower_bound = np.array([ 100, 0, 0], dtype="uint8")
upper_bound = np.array([ 255, 145, 255], dtype="uint8")

# Create a binary mask based on the color filter thresholds
mask = cv2.inRange(ycrcb_image, lower_bound, upper_bound)

# Apply the mask to the original image to obtain the filtered image
filtered_image = cv2.bitwise_and(image, image, mask=mask)
image = filtered_image.copy()
image = cv2.GaussianBlur(image,(5,5),0)

# Display the filtered image
cv2.imshow('Filtered Image', filtered_image)

lower, upper = 0,0

# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)
static_color=False

for scan in range(0,image.shape[0],(int)(image.shape[0]/15))  :
     pt1_j = (int(0), int(scan))
     pt2_j = (int(image.shape[1]), int(scan))
     color_changes=0

     for x in range (0,int(image.shape[1]), int(image.shape[1]/120)):
            # Extract the color values
            h, s, v = hsv_image[scan, x]

        # Compare the color values of adjacent pixels
            if x > pt1_j[0]:
                prev_h, prev_s, prev_v = hsv_image[scan, x - 1]
                color_distance = int(np.sqrt((h - prev_h) ** 2 + (s - prev_s) ** 2 + (v - prev_v) ** 2))
                if color_distance > 30:  
                    color_changes = color_changes+1
            
     if color_changes > 20:
          if static_color==True and lower!=0:
            continue
          elif static_color==True and lower==0: 
            upper, lower= int(scan), int(scan)
            #cv2.line(image, pt1_j, pt2_j, (0, 255, 255), 3)
            static_color=False
          else:
            #cv2.line(image, pt1_j, pt2_j, (0, 255, 255), 3)
            static_color=False
            lower = int(scan)
     else :
          static_color =True
print("upper", upper, "lower", lower)
if upper >=0 and lower>0 and upper!=lower and lower < image.shape[0]:
    image = image[upper-25:lower+25, :]
    cv2.imshow('Cropped Image', image)
                        

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the grayscale image to 160x120 pixels
#gray = cv2.resize(gray, (460, 120))

# Binarize the resized grayscale image with a threshold of 150
_, binarized = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)


# Apply Canny edge detection to the grayscale image
edges = cv2.Canny(binarized, 50, 150, apertureSize=3)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the detected contours on a separate image
contour_image = np.zeros_like(image)
cv2.drawContours(contour_image, contours, -1, (255, 0, 255), 1)

incident_pnts = []
def contour_and_line(contour, rho, theta, image, incident_pnts):
    distances = []
    for point in contour:
        x0, y0 = point[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x = a * rho
        y = b * rho
        x1 = int(x + 1000 * (-b))
        y1 = int(y + 1000 * (a))
        if y0<=(int)(min1 + (max1 - min1)/12): #and x1 <  min1 + 8 and x1 > min1 - 8:
            cv2.circle(image, (x0, (int)(min1 + (max1 - min1)/12)), 2, (0, 255, 0), 1)  # Mark the incident points on the line
            incident_pnts.append((x0, (int)(min1 + (max1 - min1)/12)))

# Display the image with the drawn contours
#cv2.imshow('Detected Contours', contour_image)


# Apply Hough line transform to detect lines in the image
#lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
#lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

cartesian_lines = []
angles = []
sum=0   

max1 = 0
max2 = 0
min1 = image.shape[0]-1
min2 = image.shape[0]-1
end = 0
rhoBlue=0
thetaBlue=0
vertical_inclination, vertical_counter=0,0

if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            angle = np.arctan2(b, a)  # angle of the line
            angles.append(angle)
            sum = sum+np.abs(angle)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            if (theta < 0.1) or (theta > np.pi - 0.1):
                 #cv2.line(cdst, pt1, pt2, (0, 255, 255), 1)
                 vertical_inclination=vertical_inclination+np.abs(angle)
                 vertical_counter = vertical_counter+1
            else:
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 1)
                #print("pt1_y ", pt1[1], " pt2_y ", pt2[1])
                
                if max1 < pt1[1]:
                  max1 = pt1[1]
                  rhoBlue = rho
                  thetaBlue = theta
                if max2 < pt2[1]:
                    max2 = pt2[1]
                    rhoBlue = rho
                    thetaBlue = theta
                if min1 > pt1[1]:
                    min1 = pt1[1]
                    rhoBlue = rho
                    thetaBlue = theta
                if min2 > pt2[1]:
                    min2 = pt2[1]
                    rhoBlue = rho
                    thetaBlue = theta
                if end < pt2[0]:
                    end = pt2[0]
                    rhoBlue = rho
                    thetaBlue = theta
        print("min1 ", min1, ",min2 ", min2, " max1 ", max1, " max2-",max2)
        cv2.line(cdst, (0, (int)(min1 + (max1 - min1)/12)), (end, (int)(min2 + (max2 - min2)/12)), (255,0,0), 1, cv2.LINE_AA)
        

    
# Iterate through contours and Hough lines to calculate distances
for i, contour in enumerate(contours):
    distance = contour_and_line(contour, rhoBlue, thetaBlue, image, incident_pnts)
avg = sum / len(lines)
avg_vertical=vertical_inclination/vertical_counter
print("media", avg)

def discardOutliersLines(lines):
# threshold for angle similarity
    angle_threshold = np.radians(2)

    # Identify and draw parallel lines
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            delta_angle = np.abs(angles[i] - angles[j])
            if delta_angle < angle_threshold and angles[i]<=avg+0.6 and angles[i]>=avg-0.6 :
                # Draw the lines on the original image
                rho_i, theta_i = lines[i][0]
                a_i = np.cos(theta_i)
                b_i = np.sin(theta_i)
                x0_i = a_i * rho_i
                y0_i = b_i * rho_i
                pt1_i = (int(x0_i + 1000 * (-b_i)), int(y0_i + 1000 * (a_i)))
                pt2_i = (int(x0_i - 1000 * (-b_i)), int(y0_i - 1000 * (a_i)))
                cv2.line(cdst, pt1_i, pt2_i, (0, 0, 255), 1)

                rho_j, theta_j = lines[j][0]
                a_j = np.cos(theta_j)
                b_j = np.sin(theta_j)
                x0_j = a_j * rho_j
                y0_j = b_j * rho_j
                pt1_j = (int(x0_j + 1000 * (-b_j)), int(y0_j + 1000 * (a_j)))
                pt2_j = (int(x0_j - 1000 * (-b_j)), int(y0_j - 1000 * (a_j)))
                cv2.line(cdst, pt1_j, pt2_j, (0, 0, 255), 1)


discardOutliersLines(lines)

# Display the image with the detected lines and valid combinations

cv2.imshow('Keyboard Location', image)
cv2.imshow('Keyboard gray', edges)
cv2.imshow("Detected Lines (in green) - Standard Hough Line Transform", cdst)

_,cols,_ = image.shape
tasti = []

print("min1 - ", min1)
for x in incident_pnts:
     # if x[1] <  min1 + 8 and x[1] > min1 - 8:
           tasti.append(x[0])
tasti = list(set(tasti))
k =0 # tasti[0]
#for i in tasti:
    #if i < k + 3 and i > k - 3:
          #tasti.remove(i)
    #else:
         #k = i
#tasti.sort()
#for i in tasti:
    #if i < k + 3 and i > k - 3:
          #tasti.remove(i)
    #k = i
for i in tasti:
    cv2.circle(image, (i, (int)(min1 + (max1 - min1)/12)), 2, (255, 0, 0), 1)
cv2.imshow("tasti", image)
print(tasti)

tasti_neri = []
tasti_bianchi =[]
k = True

for i in tasti:
    if i-2>=0 and i+2 < gray.shape[1]: 
        if gray[(int)(min1 + (max1 - min1)/12), i - 2] > 140 and gray[(int)(min1 + (max1 - min1)/12), i + 2] > 140:
            tasti_bianchi.append(i)
        else: # gray[(int)(min1 + (max1 - min1)/12), i - 1] > 160 and gray[(int)(min1 + (max1 - min1)/12), i + 1] < 60:
            tasti_neri.append(i)
    else :
         if gray[(int)(min1 + (max1 - min1)/12), i - 1] > 140 and gray[(int)(min1 + (max1 - min1)/12), i] > 140:
            tasti_bianchi.append(i)
         else: # gray[(int)(min1 + (max1 - min1)/12), i - 1] > 160 and gray[(int)(min1 + (max1 - min1)/12), i + 1] < 60:
            tasti_neri.append(i)
         

tasti_neri.sort()
k = 0
for i in tasti_neri:
    if i < k + 2 and i > k - 2:
          tasti_neri.remove(i)
    else:
         k = i


tasti_neri = list(set(tasti_neri))
print(len(tasti_bianchi), len(tasti_neri))
tasti_neri.sort()

dist = []
for i in range(1,len(tasti_neri) - 2):
      dist.append(tasti_neri[i + 1] - tasti_neri[i])
media = statistics.mean(dist)
for i in range(1,len(tasti_neri) - 2):
      if dist[i - 1] > media - 2:
            tasti_bianchi.append(tasti_neri[i] + int(dist[i - 1] / 2))

print(len(tasti_bianchi), media)
tasti_bianchi.sort()
k = 0
for i in tasti_bianchi:
    if i < k + 3 and i > k - 3:
          tasti_bianchi.remove(i)
    else:
         k = i

print(len(tasti_bianchi))

angle = avg_vertical* 180 / np.pi
#print("avg_vertical", angle)
length = 150




# for i in tasti_bianchi:
#     Px =  (int)(i + length * np.cos(angle))
#     Py =  (int)(max1 + length * np.sin(angle))
#     cv2.line(image, (i, max1), (Px, Py), (0, 0, 255), 1)
# for i in tasti_neri:
#     Px =  (int)(i + length * np.cos(angle))
#     Py =  (int)(min1 + length * np.sin(angle))
#     cv2.line(image, (i, min1), (Px, Py), (0, 255, 0), 2)
# cv2.imshow("tasti bianchi", image)

for i in tasti_bianchi:
    cv2.line(image, (i, max1), (i, min1), (0, 0, 255), 1)
for i in tasti_neri:
    cv2.line(image, (i, min1), (i, int(pt2_j[1])), (0, 255, 0), 2)
cv2.imshow("tasti bianchi", image)

# Key identification
# We cross-correlate two discrete sequences:
# the one of the keys detected with 1 if white keys and  otherwise k_u
# the one with the a valid sequence of keys k_v
k_u = []
k_v = []
for i in range(cols):
    if i in tasti_bianchi:
          k_u.append(1)
    elif i in tasti_neri:
         k_u.append(0)

k_v.append(1)
k_v.append(0)
k_v.append(1)
k_v.append(0)
k_v.append(1)
k_v.append(1)
k_v.append(1)
k_v.append(0)
k_v.append(1)
k_v.append(0)
k_v.append(1)
k_v.append(0)
k_v.append(1)
     
         
     
cv2.waitKey(0)
cv2.destroyAllWindows()