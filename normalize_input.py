from PIL import Image
import numpy as np
import pickle
dataset=np.array([0,0,0]);
index=[0,1,2]
count=0
for i in range(1,2250):
	im=Image.open("/home/hubatrix/AML_LAB/Group2_t2/scaled/"+str(i)+".jpg");
	rgb_im=im.convert('RGB')
	for j in range(0,16):
		for k in range(0,8):
			r,g,b = rgb_im.getpixel((j,k))
			data=np.array([r,g,b])
			print data
			count+=1
			dataset=np.append(dataset,[[r/255.0,g/255.0,b/255.0]])
	dataset=np.delete(dataset,index)
	pickle.dump(dataset,open("/home/hubatrix/AML_LAB/Group2_t2/data"+str(i),"wb"))
	dataset=np.array([0,0,0])
#print dataset
#print "+++++++++++++++++++"
#print count		
#pickle.dump("/home/venkat/AML_LAB/data",dataset)
