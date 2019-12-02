import numpy as np
import cv2
import random

def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	A = np.zeros((np.size(x1, axis=0) * 2, 9))
	for i in range(np.size(x1, axis=0)):
		x = x2[i, 0]
		y = x2[i, 1]
		u = x1[i, 0]
		v = x1[i, 1]

		A[2*i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u ]
		A[2*i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
	A = np.asarray(A)
	U, S, V = np.linalg.svd(A)
	H2to1 = V[-1,:]/V[-1,-1]
	H2to1 = H2to1.reshape((3,3))

	#M, mask = cv2.findHomography(x2, x1)
	return H2to1

def computeH_norm(x1, x2):
	# Q2.2.2
	# Compute the centroid of the points
	mean_x1 = np.mean(x1, axis=0)
	mean_x2 = np.mean(x2, axis=0)

	# Shift the origin of the points to the centroid
	x1 = x1 - mean_x1
	x2 = x2 - mean_x2
	distance_x1 = []
	distance_x2 = []
	for i in range(np.size(x1,axis=0)):
		distance_x1.append(np.power(x1[i,0],2))
		distance_x2.append(np.power(x2[i, 0],2))
	dis_x1_idx = np.argsort(distance_x1)[::-1]
	dis_x2_idx = np.argsort(distance_x2)[::-1]

	# Similarity transform 1
	T1 = [1 / x1[dis_x1_idx[0]][0], 0, - 1 * mean_x1[0]/ x1[dis_x1_idx[0]][0],
		0, 1 / x1[dis_x1_idx[0]][1], -1 * mean_x1[1] / x1[dis_x1_idx[0]][1],
		  0, 0, 1]
	T1 = np.array(T1)
	T1 = T1.reshape((3, 3))

	# Similarity transform 2
	T2 = [1 / x2[dis_x2_idx[0]][0], 0, - 1 * mean_x2[0] / x2[dis_x2_idx[0]][0],
		  0, 1 /x2[dis_x2_idx[0]][1], -1 * mean_x2[1] / x2[dis_x2_idx[0]][1],
		  0, 0, 1]
	T2 = np.array(T2)
	T2 = T2.reshape((3, 3))

	# Normalize the points so that the average distance from the origin is equal to sqrt(2)
	x1 /= x1[dis_x1_idx[0]]
	x2 /= x2[dis_x2_idx[0]]

	# Compute homography
	H2to1 = computeH(x1, x2)

	# Denormalization
	#print(np.linalg.lstsq(T1,H2to1)[0].shape)
	H2to1 = np.matmul(np.matmul(np.linalg.inv(T1),H2to1),T2)
	#H2to1 = np.linalg.lstsq(T1,H2to1)[0]*T2

	return H2to1

def computeH_ransac(locs1, locs2):
	# Q2.2.3
	# Compute the best fitting homography given a list of matching points
	range_num = np.size(locs1, axis=0)
	max_itr = 1000
	num_points = 4
	bestH2to1 = np.zeros((3, 3))
	pre_inliers_num = 0
	best_inliers = np.zeros((1, range_num))

	if range_num < num_points:
		inliers = best_inliers
	for i in range(0, max_itr):
		picks = random.sample(range(0, range_num), num_points)
		x1 = np.zeros((num_points, 2))
		x2 = np.zeros((num_points, 2))

		for j in range(len(picks)):
			x1[j, :] = locs1[picks[j], :]
			x2[j, :] = locs2[picks[j], :]

		H2to1 = computeH(x1, x2)

		new_x2 = locs2
		new_x2 = np.c_[new_x2, np.ones(np.size(locs2, axis=0))].T
		new_x2 = np.matmul(H2to1, new_x2)
		new_x2 = new_x2.T
		new_x2 = new_x2[:] / (new_x2[:, 2].reshape(-1, 1)+1e-9)
		new_x1 = locs1
		new_x1 = np.c_[new_x1, np.ones(np.size(locs1, axis=0))]

		diff = new_x2 - new_x1
		diff = np.sqrt(diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1])
		inliers = np.zeros((1, range_num))

		sum_diff=0
		for k in range(0, range_num):
			if diff[k] < 10:
				inliers[0,k] = 1
				sum_diff += diff[k]

		total_inliers = sum_diff

		if total_inliers > pre_inliers_num:
			best_inliers = inliers
			pre_inliers_num = total_inliers

	inliers = best_inliers
	in_1_list = np.where(inliers == 1)
	new_x2 = locs2[in_1_list[1], :]
	new_x1 = locs1[in_1_list[1], :]
	bestH2to1 = computeH_norm(new_x1, new_x2)

	return bestH2to1, inliers

def compositeH(H2to1, template, img):
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	#H2to1 = np.linalg.inv(H2to1)

	#Create mask of same size as template
	mask = np.zeros((np.size(template, axis=0), np.size(template, axis=1), np.size(template, axis=2)))

	#Warp mask by appropriate homography
	warp_mask = cv2.warpPerspective(mask, H2to1, (np.size(img, axis=1), np.size(img, axis=0)))

	#Warp template by appropriate homography
	template_mask = cv2.warpPerspective(template, H2to1, (np.size(img, axis=1), np.size(img, axis=0)))
	'''cv2.imshow("template_mask", template_mask)
	cv2.waitKey(0)'''

	composite_img = img
	for i in range(np.shape(template_mask)[0]):
		for j in range(np.shape(template_mask)[1]):
			if np.sum(template_mask[i, j, :]) == 0:
				continue
			elif np.sum(template_mask[i, j, :]) == 255 * 3:
				continue
			composite_img[i, j, :] = template_mask[i, j, :]

	return composite_img


