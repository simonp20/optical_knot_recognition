import cv2
import numpy as np
from functions import open_and_canny

# Load the image
img = cv2.imread('images/raw/figure_eight/IMG_8078.jpeg')
resized_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)

# Apply Canny edge detection to extract edges
edges = open_and_canny(img)


# Apply Hough Transform to detect curves
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

# Draw the detected circles on the original image
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)

output = np.zeros_like(img)
if circles is not None:
    circules = np.round(circles[0,:]).astype("int")
    for (x,y,r) in circles:
        cv2.circle(output, (x,y), r, (0,255,0), 2)

output[:,:,0]=edges

print(output.shape)

# Display the results
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('edge and curve', output)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Draw the edges onto the output image
# output[:,:,0] = edges
# output[:,:,1] = edges
# output[:,:,2] = edges

# # Draw the detected curves onto the output image
# if curves is not None:
#     for curve in curves:
#         x1, y1, x2, y2 = curve[0]
#         cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

# # Display the resulting image
# cv2.imshow('Edge and curve detection', output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()