# Takes a set of images as inputs, transforms them using multiple algorithms, then outputs them in CSV format

import sys
import csv
import numpy as np
import cv2

for imageDimension in [10,20,30,40]:

	outputFile = open(str(imageDimension)+"x"+str(imageDimension)+".csv",'w')

	writer = csv.writer(outputFile,delimiter=',')

	with open("image_paths.txt",'r') as file:
		lines = file.readlines()

	for line in lines:
		imagePath, label = line.split()
	#	if label != 'H':
	#		continue
		print line

		frame = cv2.imread(imagePath) # frame is a HxW numpy ndarray of triplets (pixels), where W and H are the dimensions of the input image
		frame = cv2.resize(frame,(100,100))
		  # downsize it to reduce processing time
		cv2.imshow("original",frame)
	
		###############################################################################
		# Make everything apart from the main object to be black in color
	
		converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
	
		# tuned settings
		lowerBoundary = np.array([0,40,30],dtype="uint8")
		upperBoundary = np.array([43,255,254],dtype="uint8")
		skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
		# apply a series of erosions and dilations to the mask using an elliptical kernel
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
		skinMask = cv2.erode(skinMask, kernel, iterations = 2)
		skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
	
		lowerBoundary = np.array([170,80,30],dtype="uint8")
		upperBoundary = np.array([180,255,250],dtype="uint8")
		skinMask2 = cv2.inRange(converted, lowerBoundary, upperBoundary)
	
		skinMask = cv2.addWeighted(skinMask,0.5,skinMask2,0.5,0.0)
		# blur the mask to help remove noise, then apply the
		# mask to the frame
		# skinMask = cv2.medianBlur(skinMask, 5)
		skin = cv2.bitwise_and(frame, frame, mask = skinMask)
		frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
		skin = cv2.bitwise_and(frame, frame, mask = skinMask)
	
		cv2.imshow("masked",skin) # Everything apart from skin is shown to be black
		###############################################################################
	
	
		###############################################################################
		# thresholding code
		h,w = skin.shape[:2]

		bw_image = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)  # Convert image from HSV to BGR format
		bw_image = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)  # Convert image from BGR to gray format
		bw_image = cv2.GaussianBlur(bw_image,(5,5),0)  # Highlight the main object
		threshold = 1
		for i in xrange(h):
			  for j in xrange(w):
				   if bw_image[i][j] > threshold:
				       bw_image[i][j] = 255 #Setting the skin tone to be White
				   else:
				       bw_image[i][j] = 0 #else setting it to zero.
				       
	#	cv2.imshow("thresholded",bw_image) # Skin color is shown to be completely white
		###############################################################################
	
	
		###############################################################################
		# Remove the arm by cropping the image and draw contours around the main object
		sign_image = bw_image[:h-15,:]  # Cropping 15 pixels from the bottom
		# Drawing a contour around white color.
		# 'contours' is a list of contours found.
		# 'hierarchy' is of no use as such.
		contours, hierarchy = cv2.findContours(sign_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
		# Finding the contour with the greatest area.
		largestContourIndex = 0
		if len(contours)<=0:
			print "Skipping due to empty contour"		
			continue
		largestContourArea = cv2.contourArea(contours[largestContourIndex])
		i=1
		while i<len(contours): 
			  if cv2.contourArea(contours[i]) > cv2.contourArea(contours[largestContourIndex]):
				   largestContourIndex = i
			  i+=1
		# Draw the largest contour in the image.
		cv2.drawContours(sign_image,contours,largestContourIndex,(255,255,255),thickness = -1)
		x,y,w,h = cv2.boundingRect(contours[largestContourIndex]) # Draw a rectangle around the contour perimeter
		# cv2.rectangle(sign_image,(x,y),(x+w,y+h),(255,255,255),0,8)
		###############################################################################
	
	
		#######################################################
		### centre the image in its square ###################
		squareSide = max(w,h)-1
		hHalf = (y+y+h)/2
		wHalf = (x+x+w)/2
		hMin, hMax = hHalf-squareSide/2, hHalf+squareSide/2
		wMin, wMax = wHalf-squareSide/2, wHalf+squareSide/2
	
		if (hMin>=0 and hMin<hMax and wMin>=0 and wMin<wMax):
			sign_image = sign_image[hMin:hMax,wMin:wMax]
		else:
			print "No contour found!! Skipping this image"
			continue
		
		#cv2.imshow("centred",sign_image)
		########################################################
	
		########################################################
		# finally convert the multi-dimensonal array of the
		# image to a one-dimensional one and write it to a file
		sign_image = cv2.resize(sign_image,(imageDimension,imageDimension))
	
		flattened_sign_image = sign_image.flatten() # Convert multi-dimensional array to a one-dimensional array
		outputLine = [label] + np.array(flattened_sign_image).tolist()
		writer.writerow(outputLine)
		cv2.imshow("final",sign_image)
		#########################################################
	
		if cv2.waitKey(1) & 0xFF == ord("q"): # Wait for a few microseconds and check if `q` is pressed.. if yes, then quit
			break

# cleanup the camera and close any open windows
# camera.release()
cv2.destroyAllWindows()

print "The program completed successfully !!"
