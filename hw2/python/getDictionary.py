import numpy as np
import cv2 as cv
import pickle
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from sklearn.cluster import KMeans


def get_dictionary(imgPaths, alpha, K, method):

    filterBank = create_filterbank()

    pixelResponses = np.zeros((alpha * len(imgPaths), 3 * len(filterBank)))

    for i, path in enumerate(imgPaths):
        print('-- processing %d/%d' % (i, len(imgPaths)))
        image = cv.imread('../data/%s' % path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # convert the image from bgr to rgb, OpenCV use BGR by default
        
        # -----fill in your implementation here --------
        responses = extract_filter_responses(image, filterBank)
        if method == 'random':
            points = get_random_points(image, alpha)
        else:
            k = 0.04
            points = get_harris_points(image, alpha, k)
        x = len(points)
        for l in range(0,x):
            for j in range(0,np.size(responses, axis=2)):
                pixelResponses[i * x + l, j] = responses[points[l, 1],points[l, 0], j]
        # ----------------------------------------------

    dictionary = KMeans(n_clusters=K, random_state=0).fit(pixelResponses).cluster_centers_

    return dictionary

if __name__ == "__main__":
    meta = pickle.load(open('../data/traintest.pkl', 'rb'))
    all_imagenames = meta['all_imagenames']

    alpha =50
    K = 100
    imgPaths= all_imagenames
    #print(imgPaths)
    dictionary_harris=get_dictionary(imgPaths, alpha, K, 'harris')
    filehandler_harris = open("dictionaryHarris.pkl", 'wb')
    pickle.dump(dictionary_harris, filehandler_harris)
    dictionary_random = get_dictionary(imgPaths, alpha, K, 'random')
    filehandler_random = open("dictionaryRandom.pkl", 'wb')
    pickle.dump(dictionary_random, filehandler_random)