import cv2
import numpy as np

# Load the image
img = cv2.imread('input_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection to the grayscale image
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply Hough transform to detect curves in the image
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

# Create a new blank image with the same size as the input image
output = np.zeros_like(img)

# Draw the detected curves onto the output image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Save the output image with the detected curves
cv2.imwrite('output_image.jpg', output)
