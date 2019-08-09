import cv2
import numpy as np


# //capturing web cam for video and image

cap = cv2.VideoCapture(0)

# //contin
def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# make_1080p()
make_720p()
while True:
	ret, frame = cap.read()
	# make_1080p()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow('frame' ,gray) #imag show
	# make_720p():
	cv2.imshow('frame1' ,frame) 


	frame2 = rescale_frame(frame, 	percent=75)

	#imag show
	cv2.imshow('frame2' ,frame2) #imag show
	# cv2.imshow('frame3' ,frame) #imag show
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()
