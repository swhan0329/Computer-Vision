import numpy as np
import cv2
import skimage.color
from python.helper import briefMatch
from python.helper import computeBrief
from python.helper import corner_detection

def matchPics(I1, I2, reverse=0):
	#I1, I2 : Images to match
	

	#Convert Images to GrayScale
	I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	
	#Detect Features in Both Images
	locs1= corner_detection(I1_gray,1.5)
	locs2= corner_detection(I2_gray,1.5)

	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(I1_gray, locs1)
	desc2, locs2 = computeBrief(I2_gray, locs2)

	temp_locs1,temp_locs2=[],[]
	if reverse == 1:
		for i in range(np.size(locs1,axis=0)):
			temp_locs1.append(locs1[i,0])
			locs1[i,0]=locs1[i,1]
			locs1[i,1]=temp_locs1[i]

		for j in range(np.size(locs2,axis=0)):
			temp_locs2.append(locs2[j,0])
			locs2[j,0]=locs2[j,1]
			locs2[j,1]=temp_locs2[j]

	#Match features using the descriptors
	matches = briefMatch(desc1, desc2, 0.65)

	return matches, locs1, locs2
