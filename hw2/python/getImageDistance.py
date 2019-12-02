from scipy.spatial.distance import cdist
from utils import chi2dist

def get_image_distance(hist1,hist2, method):
    if method == 'chi2':
        dist = chi2dist(hist1,hist2)
    else:
        dist = cdist(hist1,hist2)
    return  dist