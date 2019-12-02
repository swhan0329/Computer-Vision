import matplotlib.pyplot as plt
import pickle
import numpy as np
from getImageFeatures import get_image_features
from getImageDistance import get_image_distance

def evaluate_recognition_system_kNN(K, point_method, distance_metric):

    with open("vision%s.pkl" % point_method, 'rb') as file:
        dictionary = pickle.load(file)
        filterBank = pickle.load(file)
        trainFeatures = pickle.load(file)
        trainLabels = pickle.load(file)
    confusion = np.zeros((8, 8), dtype=int)
    dic_len = np.size(dictionary,axis=0)
    meta = pickle.load(open('../data/traintest.pkl', 'rb'))
    test_imagenames = meta['test_imagenames']
    test_labels = meta['test_labels']

    for i in range(0, len(test_imagenames)):
        d_chi = []
        K_count = np.zeros((1, 8), dtype=int)
        if i % 10 == 0:
            print(i, "/", len(test_imagenames))
        img_name = test_imagenames[i]
        wordMap = pickle.load(open('../data/%s_%s.pkl' % (img_name[:-4], point_method), 'rb'))

        d_chi = get_image_distance(get_image_features(wordMap, dic_len), trainFeatures, distance_metric)
        l = 0
        d_chi=np.asarray(d_chi)
        sorted_index = np.argsort(d_chi)
        sorted_d_chi = d_chi[sorted_index]

        for j in range(0, len(d_chi)):
            for k in range(0,K):
                if d_chi[j] == sorted_d_chi[k]:
                    K_count[0,int(trainLabels[j]) - 1] += 1
        c = True
        for m in range(0,8):
            K_count = K_count.flatten(-1)
            #print(K_count)
            if K_count[m] == K_count.max() and c:
                c = False
                #print(K_count)
                confusion[int(test_labels[i]) - 1, m] = confusion[int(test_labels[i]) - 1, m] + 1
        #print(confusion)

    acc, temp = 0, 0
    for i in range(0, 8):
        temp = confusion[i, i]
        acc += temp
    acc = acc / 160
    print("point method:", point_method, " accuracy:", acc)
    print(confusion)

    return acc

if __name__ == "__main__":
    acc_total = []
    K=[]

    for k in range(1, 41):
        acc = evaluate_recognition_system_kNN(k,point_method='Harris', distance_metric='chi2')
        acc_total.append(acc)
        K.append(k)

    plt.plot(K, acc_total)
    plt.xlabel('k values')
    plt.ylabel('accuracy')
    plt.show()