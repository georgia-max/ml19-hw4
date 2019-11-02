"""This module contains the function to perform principal components analysis."""
import numpy as np


def pca(data):
    """
    Perform principal components analysis to reduce dimensionality of data.

    :param data: d x n matrix of n d-dimensional data points. Each column is an example.
    :type data: ndarray
    :return: tuple containing three components: (new_data, variances, eigenvectors). The variable new_data is a d x n
    matrix containing the original data mapped to a new coordinate space. The variable variances is a length-d vector
    containing the variance captured by each new dimensions. The variable eigenvectors is a matrix where each column
    is one of the eigenvectors that the data has been projected onto.
    :rtype: tuple
    """
    #####################################################################
    # Enter your code below for computing new_data and variances.
    # You may use built in np.linalg.eig or np.linalg.svd, but you are
    # not allowed to use a pre-built pca in your implementation
    #####################################################################

    col_mean =np.mean(data, axis=1)

    center_data = data.T-col_mean
    # import matplotlib.pyplot as plt
    # plt.plot(center_data[:, 0], center_data[:, 1], '*')
    # plt.show()
    # ex=np.sum(center_data, axis=0)
    # print("this should be zero",ex)

    cov = np.cov(center_data.T)
    egi_values,eigenvectors = np.linalg.eig(cov) #V(64,) #D(64,64)
    idx= np.argsort(egi_values)[::-1]
    variances = egi_values[idx]
    eigenvectors =eigenvectors[:,idx]

    new_data = eigenvectors.T.dot(center_data.T)
    # print(new_data.T)


    #####################################################################
    # End of your contributed code
    #####################################################################

    return np.real(new_data), np.real(variances), np.real(eigenvectors)
