import cv2
import numpy as np
from os import listdir

def open_and_canny(img):
    """Performs the morphological operation opening on the image
        and performs canny edge detection

    Args:
        img (cv2_image): image to open and edge detect

    Returns:
        cv2_image: the original image that has been opened and edge
        detected
    """
    #define kernel structuring element for opening
    kernel = np.ones((5,5),np.uint8)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply opening to the grayscale image
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Perform Canny edge detection with threshold values of 100 and 200
    edges = cv2.Canny(opening, 100, 200)

    # Resize the edge-detected image to match the original image size
    edges_resized = cv2.resize(edges, (img.shape[1], img.shape[0]))
    
    return edges_resized

def detect_curves(img, ogim):
    """Detects curves on a black and white image, and writes the curves on the image as
        green and the original edges as blue

    Args:
        img (cv2 image): the cv2 image, in black and white, to detect curves upon
        
    Returns:
        cv2_image: the image with curves and edges in 2 of its color channels
    """
    #detect curves on the image
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    
    #generate output image
    output = np.zeros_like(ogim)
    if circles is not None:
        for (x,y,r) in circles:
            cv2.circle(output, (x,y), r, (0,255,0), 2)

    output[:,:,0]=img
    return output

def detect_lines(img):
    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(img, rho=1, theta=1*np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    # Draw lines on the original image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Apply Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

    # Draw circles on the original image
    if circles is not None:
        circles = np.round(circles[0, :]).astype('int')
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Detected Lines and Circles', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def warp_image(img, scale_factor, rotation_angle):
    """
	Given a cv2 image and the three factors, scales, shifts and rotates the image
    by the given factors
    
	:param file_path: file path for inputs and labels, something 
	like 'CIFAR_data_compressed/train'
	:param img: cv2 image to warp
    :param scale_factor: factor by which to scale image
    :param shift_amount: amount by which to shift the image
    :param rotation_angle: angle by which to rotate the image
	:return: the original image that has been scaled, shifted and rotated
	"""
    # Create a scaling matrix and apply it to the image
    M_scale = np.float32([[scale_factor, 0, 0], [0, scale_factor, 0]])
    scaled_img = cv2.warpAffine(img, M_scale, (img.shape[1], img.shape[0]))

    # Create a rotation matrix and apply it to the image
    M_rotate = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rotation_angle, 1)
    rotated_img = cv2.warpAffine(scaled_img, M_rotate, (img.shape[1], img.shape[0]))
    
    #return the final image
    return rotated_img


def load_images(path_to_images):
    """Load all images in the input directory and scales them to be the
        same size, 300x300 pixels

    Args:
        path_to_images (string): the path to the directory from which
        the images will be loaded

    Returns:
        list<cv2_img>: a list of all images in the directory
    """
    imgs = []
    for i in listdir(path_to_images):
        img = cv2.imread(path_to_images+'/'+i)
        resized_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)
        imgs.append(resized_img)
        
    return imgs, listdir(path_to_images)