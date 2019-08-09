import cv2
import numpy as np
import pdb
import pickle
from pyzbar import pyzbar
import argparse
# import csv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


# pdb.set_trace()
# looks for frontal face
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
print(face_cascade)
recognizer = cv2.face.LBPHFaceRecognizer_create()
# importing the trainer
recognizer.read("trainer.yml")


# importing the train labels dictionary from faces_train.py

labels = {}

with open("label.pickle","rb") as f:
	old_labels= pickle.load(f)
	#dictionary is in  {"sarat":1,"jai":2}
	#we have to convert it in {1:"sarat",2:"jai"}
	# following is the snippet
	labels= {v:k for k,v in old_labels.items()}

# cap = cv2.VideoCapture(0)

# i=0
# while (True):
	
	# ret, frame = cap.read()

	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# faces = face_cascade.detectMultiScale(gray,scaleFactor= 1.5, minNeighbors = 5)
	# # building the training model using 
	# for (x,y,w,h) in faces:
	# 	print(x,y,w,h)
	# 	roi_gray = gray[y:y+h,x:x+w]
	# 	roi_Color = frame[y:y+h,x:x+w]

		# recognize deep learning model predict keras t
		# import the necessary packages


frame = args["image"]
frame = cv2.imread(frame)
import pdb
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
pdb.set_trace()
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# ratio = 500/gray.shape[1]
# dim = (500,int(ratio * gray.shape[0]))
# gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
# cv2.imshow("gray", gray)
# cv2.imwrite("chelleng.png", gray)
# gray = cv2.resize(gray,(100,100))
# initialize the list of threshold methods
methods = [
	("THRESH_BINARY", cv2.THRESH_BINARY),
	("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV),
	("THRESH_TRUNC", cv2.THRESH_TRUNC),
	("THRESH_TOZERO", cv2.THRESH_TOZERO),
	("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV)]
	
# for (threshName, threshMethod) in methods:
	# threshold the image and show it
# for j in range(128,255):
# 	for i in range(10,128):
# 		(T, thresh) = cv2.threshold(gray, i, j, cv2.THRESH_BINARY)
# 		name = ("processed/thresh" + str(i)+str(j))+".png"
# 		cv2.imwrite(name, thresh)
# 	# if cv2.waitKey(1) & 0xFF == ord('q'):
# 	# 	break
faces = face_cascade.detectMultiScale(gray,scaleFactor= 1.3, minNeighbors = 5)
for (x,y,w,h) in faces:
	print(x,y,w,h)
	roi_gray = gray[y:y+h,x:x+w]
	# roi_Color = frame[y:y+h,x:x+w]
ratio = 500/roi_gray.shape[1]
dim = (500,int(ratio * roi_gray.shape[0]))
graynew = cv2.resize(roi_gray, dim, interpolation = cv2.INTER_AREA)
# (T, thresh) = cv2.threshold(graynew, 38, 140, cv2.THRESH_BINARY)
(T, thresh) = cv2.threshold(graynew, 45, 203, cv2.THRESH_BINARY)
# (T, thresh) = cv2.threshold(graynew, 48, 180, cv2.THRESH_BINARY)
# (T, thresh) = cv2.threshold(graynew, 37, 140, cv2.THRESH_BINARY)
cv2.imshow("gray", thresh)
cv2.imwrite("chelleng11.png", thresh)

# print(faces.shape)
id_, conf = recognizer.predict(gray)
print(id_, conf)
name = labels[id_]
print(name)

# if conf>60:
	# # if id_ == 1:
	# print("Billla Me Spotted",labels[id_])
# font = cv2.FONT_HERSHEY_SIMPLEX
# name = labels[id_]
# color = (255,255,255)
# stroke = 2
# cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
		# 	else :
		# 		print("Sarat Spotted")
		# print(id_, conf)

		# gray_face_extracted = 'MyfaceGray/{}.png'.format(i)
		# color_face_extracted = 'MyfaceColor/{}.png'.format(i)
		# cv2.imwrite(gray_face_extracted,roi_gray)
		# cv2.imwrite(color_face_extracted,roi_Color)

		# defining a rectangle in region of interface

	# color = (255, 0 ,0) #bgr
	# end_cord_x = x + w
	# end_cord_y = y + h
	# stroke =  2 # width of rectangleS
	# cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y), color, stroke)
	# # Display the resulting Frame
	# cv2.imshow('ImageFrame',frame)
	# i += 1
	# # out.write(frame)
	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	break


# cap.release()
# # out.release()
# cv2.destroyAllWindows()
