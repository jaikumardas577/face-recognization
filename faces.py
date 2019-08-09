import cv2
import numpy as np
import pdb
import pickle

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

cap = cv2.VideoCapture(0)

i=0
while (True):
	
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,scaleFactor= 1.5, minNeighbors = 5)
	# building the training model using 
	for (x,y,w,h) in faces:
		print(x,y,w,h)
		roi_gray = gray[y:y+h,x:x+w]
		roi_Color = frame[y:y+h,x:x+w]

		# recognize deep learning model predict keras t
		ratio = 500/roi_gray.shape[1]
		dim = (500,int(ratio * roi_gray.shape[0]))
		roi = cv2.resize(roi_gray, dim, interpolation = cv2.INTER_AREA)
		# (T, roi) = cv2.threshold(roi_gray, 45, 203, cv2.THRESH_BINARY)
		id_, conf = recognizer.predict(roi)
		print(id_, roi)
		if conf>50:
			# if id_ == 1:
			# print("Billla Me Spotted",labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
		# 	else :
		# 		print("Sarat Spotted")
		# print(id_, conf)

		# gray_face_extracted = 'MyfaceGray/{}.png'.format(i)
		# color_face_extracted = 'MyfaceColor/{}.png'.format(i)
		# cv2.imwrite(gray_face_extracted,roi_gray)
		# cv2.imwrite(color_face_extracted,roi_Color)

		# defining a rectangle in region of interface

		color = (255, 0 ,0) #bgr
		end_cord_x = x + w
		end_cord_y = y + h
		stroke =  2 # width of rectangleS
		cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y), color, stroke)
	# Display the resulting Frame
	cv2.imshow('ImageFrame',frame)
	i += 1
	# out.write(frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cap.release()
# out.release()
cv2.destroyAllWindows()