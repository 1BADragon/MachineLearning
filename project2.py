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
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    # Noise Test
    if False:
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
        #print "Starting 1's and 0's"
        hop_01 = copy.deepcopy(mnist.test.images)
        num_cpus = mp.cpu_count()
        pool = mp.Pool(processes=num_cpus)
        images = []
        labels = []
        #print "Sorting"
        for i,image in enumerate(hop_01):
            if mnist.test.labels[i][0] == 1 or mnist.test.labels[i][1] == 1:
                images.append(image)
                labels.append(mnist.test.labels[i])
        #print "Finished Sorting"
        images = np.array(images)
        labels = np.array(labels)       
        
        control_score = sess.run(accuracy, feed_dict={x: images, y_:labels})*100
        print control_score
        
        count = 100
        interval = 100/count
        #this is getting sloppy
        p = [c*interval for c in range(count+1)]
        
        zipped_data = []
        for t in p:
            zipped_data.append((t, images))      
        
        plorp = pool.map(make_funkier, zipped_data)
        hopfeild.set_working_weights(hopfeild_weights)
        for funky_images, percent_change in plorp:           
            pre_hop_test = sess.run(accuracy, feed_dict={x: funky_images, y_: labels}) * 100
            #print("starting recall")
            images = np.array(pool.map(hopfeild.recall, funky_images))
            # images = np.array([hopfeild.recall(hopfeild_weights, image) for image in images])
            #print("finished recall")
            hop_test = sess.run(accuracy, feed_dict={x: images, y_: labels}) * 100

            print pre_hop_test, hop_test, percent_change


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

def make_funkier(blob):
    images = blob[1]
    funky_images = copy.deepcopy(images)
    chance = blob[0]
    for _ in range(4):
        for i in range(len(funky_images)):
            for j in range(len(funky_images[i])):
                if random.random() * 100 < chance:
                    temp = funky_images[i, j] + random.random() * 2 - 1
                    if temp < 0:
                        temp = 0
                    elif temp > 1:
                        temp = 1
                    funky_images[i, j] = temp                
    percent_change = np.mean(((funky_images + 1) - (images + 1)) / (images + 1)) * 100
    return funky_images, percent_change

def make_funky(chance):
    funky_data = copy.deepcopy(mnist.test.images)
    for _ in range(2):
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


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

if __name__ == "__main__":
    main()
