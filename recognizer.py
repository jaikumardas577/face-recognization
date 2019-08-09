import cv2
import numpy as np
import pdb
import os
# pdb.set_trace()
# looks for frontal face
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
print(face_cascade)


cap = cv2.VideoCapture(0)
BASEDIR = os.getcwd()
path = os.path.join(BASEDIR,"MyfaceColor")
# path = os.path.join(path,"Jai")
path1 = os.path.join(path,"Bidyut")
path2 = os.path.join(path,"Pranjeet")
path3 = os.path.join(path,"Chelleng")
path4 = os.path.join(path,"pradipta")

print(path3)
i=0
while (True):
	
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,scaleFactor= 1.5, minNeighbors = 5)

	for (x,y,w,h) in faces:
		print(x,y,w,h)
		roi_gray = gray[y:y+h,x:x+w]
		roi_Color = frame[y:y+h,x:x+w]
		ratio = 500/roi_gray.shape[1]
		dim = (500,int(ratio * roi_gray.shape[0]))
		graynew = cv2.resize(roi_gray, dim, interpolation = cv2.INTER_AREA)
# recognizing the face using deep learning predict keras tensorflow pytorch scikit learn

		# gray_face_extracted = 'MyfaceGray/face_gray{}.png'.format(i)
		# color_face_extracted = 'MyfaceColor/color_gray{}.png'.format(i)
		gray_face_extracted = '{}.png'.format(i)
		color_face_extracted = '{}.png'.format(i)
		print(os.path.exists(path4))

		cv2.imwrite(os.path.join(path4,gray_face_extracted),roi_gray)
		# cv2.imwrite(os.path.join(path,color_face_extracted),roi_Color)
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
		breaks


cap.release()
# out.release()
cv2.destroyAllWindows()