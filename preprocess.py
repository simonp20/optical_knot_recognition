import cv2
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join
from functions import *
import random
import shutil

def preprocess(path_to_images):
    images, names = load_images(path_to_images)
    print("processing " + path_to_images)

    #todo: make 200 images from each picture. use random.choice
    scale_factors = [x / 10 for x in range(2, 10, 1)]
    rotation_angles = range(-90, 90, 10)

    training_images = []

    # Loop through each image in the directory
    for i in range(len(images)):
        img = images[i]
        edge_detected = open_and_canny(img)
        # edge_and_curve = detect_curves(edge_detected, img)
        # cv2.imshow(edge_and_curve)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        #warp images and add them to the training images list
        for j in range(200):
            scale = random.choice(scale_factors)
            rotate = random.choice(rotation_angles)
            warped_image = warp_image(edge_detected, scale, rotate)
            training_images.append(warped_image)
            
    return training_images

#initialize knots and the raw image directory
raw_directory = 'images/raw'
preprocessed_directory = 'images/preprocessed'
knots = [d for d in listdir(raw_directory) if not isfile(join(raw_directory, d))]

print("Knots to preprocess: "+' '.join(knots))

for knot in knots:
    #Delete old preprocessed images
    try:
        shutil.rmtree(preprocessed_directory + '/' + knot)
    except:
        print("New knot detected")
        
    mkdir(preprocessed_directory + '/' + knot)
    training_images = preprocess(raw_directory + '/' + knot)
    for i in range(len(training_images)):
        cv2.imwrite(preprocessed_directory + '/' + knot + '/image_' + str(i) + '.jpg', training_images[i])
