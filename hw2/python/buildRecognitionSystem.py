import pickle
import numpy as np
from createFilterBank import create_filterbank
from getImageFeatures import get_image_features
def build_recognition_system(point_method):
    #print(point_method)
    dictionary = pickle.load(open('dictionary%s.pkl' % point_method, 'rb'))
    filterBank = create_filterbank()

    meta = pickle.load(open('../data/traintest.pkl', 'rb'))
    train_imagenames = meta['train_imagenames']
    train_labels = meta['train_labels']
    K = np.size(dictionary,axis=0)
    T = np.size(train_labels,axis=0)
    trainFeatures=np.zeros((T,K))

    for i in range(0,len(train_imagenames)):
        if i % 10== 0:
            print(i, "/",len(train_imagenames))
        img_name = train_imagenames[i]
        wordMap = pickle.load(open('../data/%s_%s.pkl' % (img_name[:-4], point_method), 'rb'))
        trainFeatures[i,:]=get_image_features(wordMap,K)        
        
    with open("vision%s.pkl"% point_method, 'wb') as file:
        pickle.dump(dictionary, file)
        pickle.dump(filterBank, file)
        pickle.dump(trainFeatures, file)
        pickle.dump(train_labels, file)

if __name__ == "__main__":

    build_recognition_system(point_method='Random')
    build_recognition_system(point_method='Harris')