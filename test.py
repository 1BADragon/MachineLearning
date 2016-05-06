#!/usr/bin/env python

import numpy as np
import hopfeild
import random
from time import sleep
import Image
import copy

from tensorflow.examples.tutorials.mnist import input_data
np.set_printoptions(threshold=np.nan)

data_range = 10
data_lower = 8


def test():
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    avg = getimageavg(mnist)
    # print_data(avg)
    w = hopfeild.train(np.array(avg[data_lower:data_range]))
    img = Image.new('L', (28, 28))
    img2 = Image.new('L', (28, 28))
    pic = mnist.test.images[0]
    img.putdata((pic * 255).tolist())
    img.save("clean.png", "PNG")
    for i in range(pic.size):
        if random.randint(0, 100) < 70:
            temp = pic[i] + random.random() * 2 - 1
            if temp < 0:
                temp = 0
            elif temp > 1:
                temp = 1
            pic[i] = temp
    img.putdata((pic * 255).tolist())
    img.save("dirty.png", "PNG")



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

