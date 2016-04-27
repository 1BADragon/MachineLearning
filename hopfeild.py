import numpy as np
import copy
import Image
from time import sleep

#np.set_printoptions(threshold=np.nan)

working_weights = None

def convert_from(data):
    data = np.array(data)
    r = data.size
    s = data.shape
    data = data.flatten()
    for i in range(r):
        if data[i] > 0:
            data[i] = 1
        else:
            data[i] = 0
    data.shape = s
    return data


def convert_to(data):
    data = np.array(data)
    r = data.size
    s = data.shape
    data = data.flatten()
    for i in range(r):
        if data[i] > .5:
            data[i] = 1
        else:
            data[i] = -1
    data.shape = s
    return data


def trim(data, n=12):
    radius = int(n/2)
    data = np.delete(data, list(range(radius)), 1)
    data = np.delete(data, list(range(data.shape[1] - radius, data.shape[1])), 1)
    return data


def expand(data, n=12):
    radius = int(n/2)
    temp = np.zeros((data.shape[0], radius))
    data = np.append(temp, data, 1)
    data = np.append(data, temp, 1)
    return data


def train(data):
    data = convert_to(data)
    n = len(data)
    d = len(data[0])

    for point in data:
        if len(point) != d:
            print "Data not consistent sizes"
            exit()

    w = np.zeros((d, d))
    q = np.zeros((d, d))

    data = [np.matrix(point) for point in data]

    for i in range(n):
        w = w + np.outer(data[i], data[i])
        q = q + data[i].T.dot(data[i])



    w /= n
    q /= n
    for i in range(d):
        w[i, i] = 0
        q[i, i] = 0
    
    global working_weights
    working_weights = w
    return w

def set_working_weights(w):
    global working_weights
    working_weights = w

def recall(data, w=None, n=10):
    if w == None:
        w = copy.copy(working_weights)
    data = convert_to(data)
    order = list(range(len(data)))
    order2 = list(range(len(data)))
    for p in range(n):
        np.random.shuffle(order)
        for i in order:
            temp = np.sum(w[i, :]*data)
            if temp > 0:
                temp = 1
            else:
                temp = -1
            data[i] = temp
        if True:    #adding random elemets to "shake it up"
            np.random.shuffle(order2)
            if 6 - p > 0:
                count = 2**(6-p)
            else:
                count = 0
            for j in range(count):
                if data[order2[j]] == 1:
                    data[order2[j]] = -1
                else:
                    data[order2[j]] = 1    

    data = convert_from(data)
    return data
