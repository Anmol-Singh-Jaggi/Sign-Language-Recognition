import time
import sys
import numpy as np
import cv2
from sklearn.externals import joblib
dim = [10,20,30,40]



camera = cv2.VideoCapture(0)

def get_image():
	retval, im = camera.read()
	return im

for i in xrange(10):
	temp = get_image()
	print("Taking image...")
	camera_capture = get_image()
	cv2.imshow("win",camera_capture)
	time.sleep(2)
		
	file = "/home/test_image"+str(i)+".png"
	if camera_capture.any()!=0:
		print "Its fucked up"	
	else:
		cv2.imwrite(file, camera_capture)

del(camera)

print "The program completed successfully !!"
