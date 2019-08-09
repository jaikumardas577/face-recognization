import os



# changing the name of the existing file 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"MyfaceColor")
sarat_dir = os.path.join(image_dir,"Sarat")
jai_dir = os.path.join(image_dir,"jai")
bidyut_dir = os.path.join(image_dir,"Bidyut")
# bidyut_dir = os.path.join(image_dir,"Bidyut")
pranjeet_dir = os.path.join(image_dir,"Pranjeet")
chelleng_dir = os.path.join(image_dir,"Chelleng")




for root1,dirs1,files1 in os.walk(bidyut_dir) :
	i=1
	for file in files1:
		try:
			print(os.path.join(bidyut_dir, file)," ",os.path.join(bidyut_dir, str(i)+'.png'))
			os.replace(os.path.join(bidyut_dir, file), os.path.join(bidyut_dir, str(i)+'.png'))
			print(" ",os.replace(os.path.join(bidyut_dir, file), os.path.join(bidyut_dir, str(i)+'.png')))
			i = i+1
		except:
			i =  i+1


# i=1
# for root1,dirs1,files1 in os.walk(sarat_dir) :
# 	for file in files1:
# 	    os.replace(os.path.join(jai_dir, file), os.path.join(jai_dir, str(i)+'.png'))
# 	    i = i+1