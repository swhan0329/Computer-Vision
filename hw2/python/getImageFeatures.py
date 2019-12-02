import numpy as np
import pickle
import numpy as np
import cv2 as cv
from createFilterBank import create_filterbank
from getVisualWords import get_visual_words

def get_image_features(wordMap, dictionarySize):

    # -----fill in your implementation here --------
    vec = wordMap.flatten(1)
    #print(vec)
    h = np.zeros((1,dictionarySize))

    for i in range(0, dictionarySize):
        sum = 0
        for j in range(0, len(vec)):
            if round(vec[j])==i:
                sum = sum + i
        h[0,i] = sum
    for i in range(0, dictionarySize):
        h[0,i] = h[0,i] / h.sum(axis=1)
    #print(h)
    # ----------------------------------------------
    
    return h

if __name__ == "__main__":
    meta = pickle.load(open('../data/traintest.pkl', 'rb'))
    all_imagenames = meta['all_imagenames']

    imgPaths= all_imagenames
    point_method = 'Harris'
    dictionary = pickle.load(open('dictionary%s.pkl' % point_method, 'rb'))
    filterBank = create_filterbank()

    #for i in range(5,10):
    for i in range(0,len(all_imagenames)):
        if i % 10 == 0:
            print(i,"/",len(all_imagenames))
        img_name = all_imagenames[i]
        image = cv.imread('../data/%s' % img_name)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # convert the image from bgr to rgb
        wordMap = pickle.load(open('../data/%s_%s.pkl' % (img_name[:-4], point_method), 'rb'))
        get_image_features(wordMap,len(dictionary))