#!/usr/bin/python

# PCA portion of my final project

from numpy import matlib
from numpy import matrix
import fileinput
import numpy as np


def pca(matrix_data, numRow, numCol, k=10):
    # print(matrixData)
    matrix_data = matrix(matrix_data)
    orginal_shape = matrix_data.shape
    try:
        matrix_data.shape = (numRow, numCol)
    except ValueError:
        print matrix_data.size, numRow, numCol
        exit(-1)
    column_means = matrix_data.mean(0)
    
    centered_data = matrix_data.copy()
    
    (num_row, num_col) = centered_data.shape
    
    # print(num_row, num_col)

    for i in range(num_col):
      for j in range(num_row):
        centered_data[j,i] = centered_data[j,i] - column_means[0,i]

    # print(centered_data)
    
    c = centered_data.getT().dot(centered_data)
    c = c/num_row
    # print c
    
    c_e, c_v = np.linalg.eig(c)
    idx = c_e.argsort()[::-1]
    c_e = c_e[idx]
    c_v = c_v[:, idx]
    c_v = c_v.getT()
    
    # print c_e
    # print c_v

    if c_e.size > k:
        c_e = c_e[0:k]
        c_v = c_v[0:k, :]

    # print c_e
    # print c_v

    compressed_data = centered_data.dot(c_v.getT())

    #print compressed_data

    return compressed_data.A1

def pcafrominput():
    data = [line for line in fileinput.input()]

    matrixDim = data[0].split(' ')
    matrixDim = (int(matrixDim[0]), int(matrixDim[1]))

    matrixData = data[1:]
    matrixData = [line.split() for line in matrixData]
    matrixData = matlib.matrix(matrixData).astype(np.float);
    
    pca(matrixData)
    
    

    
if __name__ == '__main__':
    pcafrominput()