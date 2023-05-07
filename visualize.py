import os
import random
import cv2

# set directory path and number of images to display
dir_path = "images/preprocessed/figure_eight"
num_images = 5

# get list of image file names in directory
img_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg') or f.endswith('.png')]

# select a random sample of images
selected_imgs = random.sample(img_files, num_images)

# create window to display images
cv2.namedWindow('Random Images', cv2.WINDOW_NORMAL)

# loop through selected images and display in window
for img in selected_imgs:
    # read image and resize for display
    img_path = os.path.join(dir_path, img)
    img_disp = cv2.imread(img_path)
    img_disp = cv2.resize(img_disp, (500, 500))

    # display image in window
    cv2.imshow('Random Images', img_disp)
    cv2.waitKey(0)

# close window when done
cv2.destroyAllWindows()
