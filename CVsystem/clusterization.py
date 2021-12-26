import numpy as np

def kmeans(data, k=3, distance_method='euclidean'):
    data = data.T





    k = min(k, data.shape[0])

    if type(distance_method) is str and distance_method == 'manhattan':
        distance_method = 'cityblock'

    dist = distance.pdist(data, distance_method)
    dist = distance.squareform(dist)
    dist = np.sort(dist, 0)[1:]

    return dist[k-1] > outlier_factor

if __name__ == '__main__':
    data_lin = np.array([4,4, 3,5, 1,2, 5,5, 0,1, 2,2, 4,5, 2,1])
    data = data_lin.reshape([-1, 2]).T
