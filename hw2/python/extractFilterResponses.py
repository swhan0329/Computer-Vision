import cv2 as cv
from RGB2Lab import rgb2lab
from utils import *
import createFilterBank
def extract_filter_responses(I, filterBank):

    I = I.astype(np.float64)
    if len(I.shape) == 2:
        I = np.tile(I, (3, 1, 1))

    # -----fill in your implementation here --------
    L = rgb2lab(I)
    h = np.size(I,0)
    w = np.size(I,1)
    k = 0
    filterResponses = np.zeros((h,w,3*len(filterBank)),dtype=int) # fix 60
    #print(filterResponses.shape)
    #print(filterbank[0])
    for i in range(0,len(filterBank)):
        filterResponses[:,:,k] = imfilter(L[:,:,0], filterBank[i])
        filterResponses[:, :, k+1] = imfilter(L[:, :, 1], filterBank[i])
        filterResponses[:, :, k+2] = imfilter(L[:, :, 2], filterBank[i])
        k = k+3
        '''
        cv.imshow("filterResponses %d-th image" % (i), imfilter(L[:, :, 0], filterBank[i]))
        cv.imshow("filterResponses %d-th image" % (i), imfilter(L[:, :, 1], filterBank[i]))
        cv.imshow("filterResponses %d-th image" % (i), imfilter(L[:, :, 2], filterBank[i]))
        cv.waitKey(0)
        '''
    # ----------------------------------------------
    
    return filterResponses

if __name__ == "__main__":
    img = cv.imread('../data/airport/sun_aerinlrdodkqnypz.jpg')
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    filterbank = createFilterBank.create_filterbank()
    extract_filter_responses(img,filterbank)
