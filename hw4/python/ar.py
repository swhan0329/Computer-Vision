import os

#Import necessary functions
from python.planarH import *
from python.matchPics import *
from python.loadVid import *

#Write script for Q3.1
cv_cover = cv2.imread('../data/cv_cover.jpg')
ar_source = '../data/ar_source.mov'
book = '../data/book.mov'

ar_source_frame = loadVid(ar_source)
book_frame = loadVid(book)

ar_w_center, ar_h_center = ar_source_frame.shape[2]/2, ar_source_frame.shape[1]/2
ar_black = ar_source_frame.shape[1]/7
crop_w, crop_h = cv_cover.shape[1]/2, cv_cover.shape[1]/2

fcc = cv2.VideoWriter_fourcc(*"MJPG")
if not os.path.exists("../result"):
    os.makedirs("../result")
out = cv2.VideoWriter('../result/ar.avi', fcc, 25.0, (640, 480), True)

save_frame = []
for f in range(len(ar_source_frame)):
    if f % 20 == 0:
        print("{0}/{1}".format(f,len(ar_source_frame)))
    a_frame = ar_source_frame[f]
    b_frame = book_frame[f]
    matches, locs1, locs2 = matchPics(cv_cover,b_frame,1)

    new_locs1 = np.zeros((np.size(matches,axis=0),2))
    new_locs2 = np.zeros((np.size(matches,axis=0),2))

    for i in range(0,np.size(matches,axis=0)):
        new_locs1[i, :] = locs1[matches[i,0], :]
        new_locs2[i, :] = locs2[matches[i, 1], :]

    bestH2to1, inliers = computeH_ransac(new_locs2,new_locs1)
    crop_ar_frame = a_frame[int(ar_h_center-2.5*ar_black-5):int(ar_h_center+2.5*ar_black+5),int(ar_w_center-crop_w):int(ar_w_center+crop_w)]

    crop_ar_frame = cv2.resize(crop_ar_frame, dsize=(350,440), interpolation=cv2.INTER_LINEAR)
    composite_img = compositeH(bestH2to1,crop_ar_frame,b_frame)
    save_frame.append(composite_img)

# Write the frame into the file 'output.avi'
for ff in range(len(ar_source_frame)):
    out.write(save_frame[ff])
out.release()

