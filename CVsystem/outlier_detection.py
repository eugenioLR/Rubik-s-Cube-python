import numpy as np
from scipy.spatial import distance

def box_and_whisker(data, outlier_factor=1.25):
    """
    1 variable method
    """

    Q1 = np.quantile(data, 0.25)
    Q3 = np.quantile(data, 0.75)

    range = [Q1 - outlier_factor*(Q3-Q1), Q3 + outlier_factor*(Q3-Q1)]

    return np.logical_and(data < range[0], data > range[1])

def standard_deviation(data, outlier_factor=2):
    """
    1 variable method
    """

    x_mean = data.mean()

    x_std = np.sqrt(((data - x_mean)**2).mean())

    range = [x_mean - outlier_factor*x_std, x_mean + outlier_factor*x_std]

    return np.logical_and(data < range[0], data > range[1])


def regression(data, outlier_factor=2):
    """
    2 variable method, the array must be of shape 2xN
    where N is the number of data points
    """

    data_x = data[0, :]
    data_y = data[1, :]

    x_mean = data_x.mean()
    y_mean = data_y.mean()

    Sxy = ((data_x*data_y).mean()) - x_mean*y_mean
    Sx2 = data_x.var()

    b = Sxy/Sx2
    a = y_mean-b*x_mean

    data_yc = np.array(list(map(lambda x: b*x+a, data_x)))
    Sr = np.sqrt(((data_y - data_yc)**2).mean())

    return np.abs(data_y-data_yc) > outlier_factor*Sr

def kmeans(data, k=3, outlier_factor=2.5, distance_method='euclidean'):
    data = data.T
    k = min(k, data.shape[0])

    if type(distance_method) is str and distance_method == 'manhattan':
        distance_method = 'cityblock'

    dist = distance.pdist(data, distance_method)
    dist = distance.squareform(dist)
    dist = np.sort(dist, 0)[1:]

    return dist[k-1] > outlier_factor

def RDLOF_decision(data, outlier_factor=0.50):
        """
        1 variable method
        """

        Q1 = np.quantile(data, 0.25)
        Q3 = np.quantile(data, 0.75)

        range = [Q1 - outlier_factor*(Q3-Q1), Q3 + outlier_factor*(Q3-Q1)]

        return data < range[0]


def relative_density(data, k=3, outlier_factor=0.50, distance_method='euclidean'):
    data = data.T

    k = min(k, data.shape[0]-1)

    if type(distance_method) is str and distance_method == 'manhattan':
        distance_method = 'cityblock'

    if k <= 2:
        return np.zeros(data.shape[0]) == 0
    else:
        # Distance computation
        dist = distance.pdist(data, distance_method)
        dist = distance.squareform(dist)

        # Sorting of distances
        dist_sorted = np.sort(dist, 0)[1:]
        order = np.argsort(dist, 0)[1:]

        # Calculation of cardinality
        last_values = dist_sorted[k-1, :]
        card = (k-1) * np.ones(data.shape[0])
        card = card.astype(np.int32)
        repetitions = np.count_nonzero(dist==last_values, axis=0)
        card = card + repetitions

        # Calculation of density
        dens = np.zeros(data.shape[0])
        for i in range(len(card)):
            dens[i] = card[i]/dist_sorted[:card[i], i].sum()

        # Calculation of relative density
        drm = np.zeros(dens.shape)
        for i in range(len(drm)):
            dens_sum = dens[order[:card[i],i].T]
            drm[i] = dens[i]/(dens_sum.sum()/card[i])

        #return RDLOF_decision(drm, outlier_factor)
        return drm < outlier_factor



if __name__ == '__main__':
    print("TESTS\n")

    data_lin = np.array([3,2, 3.5,12, 4.7,4.1, 5.2,4.9, 7.1,6.1, 6.2,5.2, 14,5.3])
    data = data_lin.reshape([-1, 2]).T
    data_x = data[0,:]
    data_y = data[1,:]
    print(data)
    print()

    result = box_and_whisker(data_x, 1.25)
    print("Regression method")
    print("Outliers: ")
    print(data_x[result])
    print()

    result = standard_deviation(data_y, 2)
    print("Regression method")
    print("Outliers: ")
    print(data_y[result])
    print()

    result = regression(data, 2)
    print("Regression method")
    print("Outliers: ")
    print(data[:, result])
    print()

    data2_lin = np.array([4,4, 4,3, 5,5, 1,1, 5,4])
    data2 = data2_lin.reshape([-1, 2]).T
    print(data2)
    print()

    result = kmeans(data2, 3, 2.5, 'euclidean')
    print("Kmeans method")
    print("Outliers: ")
    print(data2[:, result])
    print()

    #result = realative_density(data2, 3, 0.5, 'manhattan')
    result = relative_density(data2, 3, 0.5, 'manhattan')
    print("Relative density method")
    print("Outliers: ")
    print(data2[:, result])
    print()
