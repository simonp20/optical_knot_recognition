import cv2

# Load the image
img = cv2.imread('images/Figure-Eight.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform Canny edge detection with threshold values of 100 and 200
edges = cv2.Canny(gray, 300, 600)

# Display the original image and the edge-detected image
cv2.imshow('Original Image', img)
cv2.imshow('Edge-Detected Image', edges)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
