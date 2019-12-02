import numpy as np
import cv2 as cv
import pickle
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses
from createFilterBank import  create_filterbank
from skimage.color import label2rgb

def get_visual_words(I, dictionary, filterBank):

    # -----fill in your implementation here --------
    responses = extract_filter_responses(I, filterBank)
    h = np.size(I, 0)
    w = np.size(I, 1)

    wordMap = np.zeros((h,w))
    for hh in range(h):
        for ww in range(w):
            arr_filter = np.asarray([responses[hh][ww][n] for n in range(np.size(responses,axis=2))])

            D = cdist(dictionary, [arr_filter],metric='euclidean')
            index_min = np.argmin(D)
            wordMap[hh][ww] = index_min

    # ----------------------------------------------
    return wordMap

if __name__ == "__main__":
    meta = pickle.load(open('../data/traintest.pkl', 'rb'))
    all_imagenames = meta['all_imagenames']

    imgPaths= all_imagenames
    point_method = 'Harris'
    dictionary = pickle.load(open('dictionary%s.pkl' % point_method, 'rb'))
    filterBank = create_filterbank()

    '''    for i in range(0,3):
    #for i in range(0,len(all_imagenames)):
        if i % 10 == 0:
            print(i,"/",len(all_imagenames))
        img_name = all_imagenames[i]
        print(img_name)
        image = cv.imread('../data/%s' % img_name)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # convert the image from bgr to rgb
        wordMap = get_visual_words(image, dictionary, filterBank)
        pickle.dump(wordMap, open('../data/%s_%s.pkl' % (img_name[:-4], point_method), 'wb'))
        img = label2rgb(wordMap)
        cv.imshow("Visual Words_HR", img)
        cv.waitKey(0)'''

    all_imagenames = meta['all_imagenames']


    point_method = 'Random'
    dictionary = pickle.load(open('dictionary%s.pkl' % point_method, 'rb'))
    for i in range(2, 3):
    #for i in range(0, len(all_imagenames)):
        if i % 10 == 0:
            print(i, "/", len(all_imagenames))
        img_name = all_imagenames[i]
        print(img_name)
        image = cv.imread('../data/%s' % img_name)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # convert the image from bgr to rgb
        wordMap = get_visual_words(image, dictionary, filterBank)
        pickle.dump(wordMap, open('../data/%s_%s.pkl' % (img_name[:-4], point_method), 'wb'))
        img = label2rgb(wordMap)
        cv.imshow("Visual Words_RD", img)
        cv.waitKey(0)