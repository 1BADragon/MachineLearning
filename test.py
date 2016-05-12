#!/usr/bin/env python

import numpy as np
import hopfeild
import random
from time import sleep
import Image
import copy

from tensorflow.examples.tutorials.mnist import input_data
np.set_printoptions(threshold=np.nan)

def test():
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    testimg = blackandwhite(mnist.test.images[0])

    img = Image.new('L', (28, 28))
    img.putdata((testimg*255).tolist())
    img.show()

    testimg.shape = (28, 28)
    pixelVal = 0
    while(pixelVal == 0):
        x = random.randint(0, 27)
        y = random.randint(0, 27)
        pixelVal = testimg[x, y]

    v = getUnitVector((random.random()*2-1, random.random()*2-1))

    for i in range(20):
        x = int(x + i*v[0])
        y = int(y + i*v[1])
        if 0 <= y < 28 and 28 > x >= 0:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    xtemp = x + i;
                    ytemp = y + j;
                    if xtemp > 27:
                        xtemp = 27
                    elif xtemp < 0:
                        xtemp = 0;
                    elif ytemp > 27:
                        ytemp = 27
                    elif ytemp < 0:
                        ytemp = 0
                    testimg[xtemp, ytemp] = 1

    testimg.shape = (testimg.size,)
    img.putdata((testimg * 255).tolist())
    img.show()

def getUnitVector(v):
    v = list(v)
    mag = 0
    for i in v:
        mag += i**2
    mag = np.sqrt(mag)
    for i, j in enumerate(v):
        v[i] = j/mag
    return tuple(v)


def blackandwhite(img):
    for i, val in enumerate(img):
        if val < .5:
            img[i] = 0
        else:
            img[i] = 1
    return img


def getimageavg(mnist):
    """
    This function is used to get the "average" of all handwriting samples
    of the same type.
    :return: list of image "averages" in numerical order
    """
    images = mnist.test.images
    labels = mnist.test.labels

    image_average = []
    for number in range(10):
        pictures = []

        for i in range(len(labels)):
            if labels[i][number] == 1.:
                pictures.append(images[i])

        average = pictures[0]

        for i in range(1, len(pictures)):
            average = np.add(average, pictures[i])

        average /= len(pictures)

        for i in range(len(average)):
            if average[i] >= .5:
                average[i] = 1
            else:
                average[i] = 0

        image_average.append(average)

    return image_average


def print_data(avg):
    fout = open("output", "w")
    for point in avg:
        for (i, item) in enumerate(point.tolist()):
            if i % 28 == 0 and i != 0:
                fout.write('\n')
            fout.write(str(item) + ' ')
        fout.write('\n')
    exit()

if __name__ == "__main__":
    test()

