import pickle
import numpy as np
from getImageFeatures import get_image_features
from getImageDistance import get_image_distance
def evaluate_recognition_system_NN(point_method, distance_metric):
    print("point method:",point_method," distance metric:",distance_metric)
    with open("vision%s.pkl"% point_method, 'rb') as file:
        dictionary= pickle.load(file)
        filterBank = pickle.load(file)
        trainFeatures= pickle.load(file)
        trainLabels = pickle.load(file)
    confusion = np.zeros((8,8),dtype=int)

    meta = pickle.load(open('../data/traintest.pkl', 'rb'))
    test_imagenames = meta['test_imagenames']
    test_labels = meta['test_labels']
    K = np.size(dictionary,axis=0)

    for i in range(0,len(test_imagenames)):
        d_chi = []
        if i % 10 == 0:
            print(i,"/",len(test_imagenames))
        img_name = test_imagenames[i]
        wordMap = pickle.load(open('../data/%s_%s.pkl' % (img_name[:-4], point_method), 'rb'))
        if distance_metric == 'chi2':
            d_chi = get_image_distance(get_image_features(wordMap, K), trainFeatures, distance_metric)
            l = 0

            for j in range(0, len(d_chi)):
                if d_chi[j] == min(d_chi):
                    l = j
                    confusion[int(test_labels[i])-1,int(trainLabels[l])-1] = confusion[int(test_labels[i])-1,int(trainLabels[l])-1] +1
            #print(confusion)
        else:
            d = get_image_distance(get_image_features(wordMap,K),trainFeatures, distance_metric)
            l = 0
            d=d.flatten(-1)
            for j in range(0, len(d)):
                if d[j] == d.min():
                    l = j
                    confusion[int(test_labels[i])-1,int(trainLabels[l])-1] = confusion[int(test_labels[i])-1,int(trainLabels[l])-1] +1

            #print(confusion)
    acc, temp = 0, 0
    for i in range(0,8):
        temp = confusion[i,i]
        acc += temp
    acc = acc / 160
    print("point method:", point_method," distance metric:",distance_metric, " accuracy:", acc)
    print(confusion)
    return acc
if __name__ == "__main__":

    evaluate_recognition_system_NN(point_method='Random', distance_metric='euclidean')
    evaluate_recognition_system_NN(point_method='Harris', distance_metric='euclidean')
    evaluate_recognition_system_NN(point_method='Random', distance_metric='chi2')
    evaluate_recognition_system_NN(point_method='Harris', distance_metric='chi2')