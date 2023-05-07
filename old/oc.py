import cv2
import numpy as np

# Load the image
img = cv2.imread('images/Figure-Eight.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the structuring element for opening
kernel = np.ones((9,9),np.uint8)

# Apply opening to the grayscale image
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

# Perform Canny edge detection with threshold values of 100 and 200
edges = cv2.Canny(opening, 300, 600)
edges2 = cv2.Canny(opening, 200, 400)

# Display the original image, the opened image, and the edge-detected image
cv2.imshow('Original Image', img)
cv2.imshow('Opened Image', opening)
cv2.imshow('edge-det lower', edges2)
cv2.imshow('Edge-Detected Image', edges)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
