#!/usr/bin/env python
from _lsprof import profiler_entry

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import random
import copy
import multiprocessing as mp
import Image

import hopfeild

change_chance = 40

l = np.array([[0.]*10]*10)

for i in range(10):
    l[i, i] = 1.

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

def main():
    """
    Main
    """
    pixelManipRange = .2

    image_averages = np.array(getimageavg(mnist))

    hopfeild_weights = hopfeild.train(np.array(image_averages[0:2]))

    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))

    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, w) + b)

    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    clean_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100

    avg_test = sess.run(accuracy, feed_dict={x: image_averages, y_: l}) * 100

    print clean_test, avg_test

    count = 100
    interval = 100/count
    p = [r*interval for r in range(count)]
    num_cpus = mp.cpu_count()

    pool = mp.Pool(processes=num_cpus)

    t = pool.map(make_funky, p)

    for funky_data, percent_change in t:
        dirty_test = sess.run(accuracy, feed_dict={x: funky_data, y_: mnist.test.labels}) * 100
        print dirty_test, percent_change

    # testing the hopfeild with 0's and 1's
    if False:
        hop_01 = copy.deepcopy(mnist.test.images)

        images = []
        labels = []
        for i,image in enumerate(hop_01):
            if mnist.test.labels[i][0] == 1 or mnist.test.labels[i][1] == 1:
                images.append(image)
                labels.append(mnist.test.labels[i])

        images = np.array(images)
        labels = np.array(labels)

        pre_hop_test = sess.run(accuracy, feed_dict={x: images, y_: labels}) * 100
        print("starting recall")
        images = np.array([hopfeild.recall(hopfeild_weights, image) for image in images])
        print("finished recall")
        hop_test = sess.run(accuracy, feed_dict={x: images, y_: labels}) * 100

        print pre_hop_test, hop_test


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


def make_funky(chance):
    funky_data = copy.deepcopy(mnist.test.images)
    for i in range(len(funky_data)):
        for j in range(len(funky_data[i])):
            if random.randint(0, 100) < chance:
                temp = funky_data[i, j] + random.random() * 2 - 1
                if temp < 0:
                    temp = 0
                elif temp > 1:
                    temp = 1
                funky_data[i, j] = temp
    percent_change = np.mean(((funky_data + 1) - (mnist.test.images + 1)) / (mnist.test.images + 1)) * 100
    return funky_data, percent_change

if __name__ == "__main__":
    main()