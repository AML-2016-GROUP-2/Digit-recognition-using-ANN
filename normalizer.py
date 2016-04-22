from PIL import Image
for i in range(1,3000):
	sample=Image.open("/home/venkat/AML_LAB/images/"+str(i)+".jpg");
	sample=sample.resize((16,8),Image.ANTIALIAS);
	sample.save("/home/venkat/AML_LAB/scaled/"+str(i)+".jpg",quality=95);
