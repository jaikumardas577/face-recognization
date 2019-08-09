import cv2
import argparse
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

image =  cv2.imread(args["name"])
cv2.imshow("Actual Image", image)


# manipulation of image


image_new = cv2.resize(image,(300,300))