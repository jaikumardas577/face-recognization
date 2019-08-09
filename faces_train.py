import os
import glob
from PIL import Image
import pickle
import numpy as np
import cv2
import inspect

# we already have find roi in the dataset still we are choosing it ,i dont why fuck it i need coffee 

# looks for frontal face
# building the recognizer
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')


# building the training model using 
recognizer = cv2.face.LBPHFaceRecognizer_create()

# changing the name of the existing file 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"MyfaceColor")
# sarat_dir = os.path.join(image_dir,"Sarat")
# pranjeet_dir = os.path.join(image_dir,"Pranjeet")
# jai_dir = os.path.join(image_dir,"jai")

# i=1
# for root1,dirs1,files1 in os.walk(jai_dir) :
# 	for file in files1:
# 	    os.rename(os.path.join(jai_dir, file), os.path.join(jai_dir, str(i)+'.png'))
# 	    i = i+1


current_id = 1

# initially taken as empty
y_labels =[]
x_train = []
y_train = []


labels_id = {}

for root,dirs,files in os.walk(image_dir):
	# print(os.path,"path")
	# print(root,"root are")
	# print(dirs,"dir are")
	# print(files,"files are")


	for file in files:
		if file.endswith('png') or file.endswith('jpg'):
			path = os.path.join(root,file)
			label = os.path.basename(root).replace("  ","_",).lower()
# explaining the steps
			# if label in labels_id:
			# 	pass
			# else :
			# 	labels_id[label] = current_id
			# 	current_id += 1
# writting commented steps in efficient way
			if not label in labels_id:
				labels_id[label] = current_id
				current_id += 1
			id_ = labels_id[label]
			print(label,path)
			print(path)

			# we cant train a image path so
			# importing image using pillow
			pil_image = Image.open(path).convert("L") #grayscale
			# will return an Image object with properties like show(),height(),width() etc

			size=(750,750)
			final_image = pil_image.resize(size,Image.ANTIALIAS)
			pil_image.resize(size)

# inspecting a python object Pil to look into its method
			# print(inspect.getmembers(pil_image, predicate=inspect.ismethod))

			image_array = np.array(pil_image,"uint8")


			# identifying the friontal image in the given image
			faces = face_cascade.detectMultiScale(image_array)
			for x,y,w,h in faces:
				roi = image_array[y:y+h,x:x+w]
				# (T, roi) = cv2.threshold(roi, 45, 203, cv2.THRESH_BINARY)
				x_train.append(roi)
				y_train.append(id_)
# print(labels_id) 
print(x_train)
y_train = np.array(y_train) 
print(y_train)

with open("label.pickle","wb") as f:
	pickle.dump(labels_id, f)


recognizer.train(x_train,y_train)
recognizer.save("trainer.yml")
