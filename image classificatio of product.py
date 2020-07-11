from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

FLAGS = None

import os
import glob
import cv2
import pickle
from progressbar import ProgressBar
import numpy as np

def mymodel(x):
    
    x_image = tf.reshape(x, [-1, 32, 32, 3])
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second layer.
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 3])
    b_fc2 = bias_variable([3])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob

def conv2d(x, W):
    
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def my_data(folder):

    pixel = 32
    if os.path.isfile('training-data.pkl') == False:
        imgnames = []
        for fname in glob.glob(folder+'*.jpg'):
            imgnames.append(fname)
        
        labels = []
        data = []
        pbar = ProgressBar()
        for i in pbar(range(len(imgnames))):
            img = cv2.imread(imgnames[i])
            if img == None:
                continue
            labels.append(int(imgnames[i][7])-1)
            
            img_small = cv2.resize(img, (pixel,pixel), \
                                   interpolation = cv2.INTER_AREA)
           
            data.append(img_small.reshape(-1))
            
            
            labels.append(int(imgnames[i][7])-1)
            flip_img = cv2.flip(img,1)
            flip_img = cv2.resize(flip_img, (pixel,pixel), \
                                  interpolation = cv2.INTER_AREA)
            data.append(flip_img.reshape(-1))
            
            
            labels.append(int(imgnames[i][7])-1)
            noise = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
            m = (50,50,50)
            s = (50,50,50)
            cv2.randn(noise,m,s)
            noise_img = img + noise
            noise_img = cv2.resize(noise_img, (pixel,pixel), \
                                   interpolation = cv2.INTER_AREA)
            data.append(noise_img.reshape(-1))
            
            labels.append(int(imgnames[i][7])-1)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            contrast_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            contrast_img = cv2.resize(contrast_img, (pixel,pixel), \
                                      interpolation = cv2.INTER_AREA)
            data.append(contrast_img.reshape(-1))
            
            
            labels.append(int(imgnames[i][7])-1)
            crop_img = img[int(0.1*img.shape[0]):img.shape[0]-int(0.1*img.shape[0]) \
                            ,int(0.1*img.shape[1]) :img.shape[1]-int(0.1*img.shape[0]), :]
            crop_img = cv2.resize(crop_img, (pixel,pixel), \
                                   interpolation = cv2.INTER_AREA)
            data.append(crop_img.reshape(-1))
            
            
            labels.append(int(imgnames[i][7])-1)
            M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),20,1)
            rot_img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
            rot_img = cv2.resize(rot_img, (pixel,pixel), \
                                   interpolation = cv2.INTER_AREA)
            data.append(rot_img.reshape(-1))
                
        
        mean = np.mean(data, axis=0)
        sd = np.std(data, axis = 0)
        data = np.asarray(data, dtype = np.float32)
        data -= mean
        data /= sd
        
        #data labels one-hot labels
        classes = 5
        labels = np.asarray(labels, dtype = np.int32)
        one_hot = np.zeros((labels.shape[0], classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
                
        with open('training-data.pkl', 'wb') as f:
            pickle.dump([data, labels, one_hot, mean, sd], f)
    
    else:
        with open('training-data.pkl', 'rb') as f:
            data, labels, one_hot, mean, sd = pickle.load(f)
    
    print('Number of images in class-1: {}'.format((labels == 0).sum()))
    print('Number of images in class-2: {}'.format((labels == 1).sum()))
    print('Number of images in class-3: {}'.format((labels == 2).sum()))
    
    
    assert data.shape[0] == one_hot.shape[0]
    total = data.shape[0]
    
    
    temp = zip(data,one_hot)
    np.random.shuffle(temp)
    data, one_hot = zip(*temp)    
    data = np.asarray(data, dtype = np.float32)
    one_hot = np.asarray(one_hot, dtype = np.int32)    
    
    
    trainX = data[:int(round(total*0.75)),:]
    trainY = one_hot[:int(round(total*0.75))]
    testX = data[int(round(total*0.75)):,:]
    testY = one_hot[int(round(total*0.75)):]
    
    
    assert trainX.shape[0] == trainY.shape[0]
    assert testX.shape[0] == testY.shape[0]
    
    print('Size of training set: {}'.format(trainX.shape[0]))
    print('Size of test set: {}'.format(testX.shape[0]))
    
    return trainX, trainY, testX, testY
    
data_index = 0
def generate_batch(X, y, batch_size):
    global data_index
    
    if data_index >= X.shape[0]:
        data_index = 0
    batch = tuple([X[data_index:data_index+batch_size,:], y[data_index:data_index+batch_size]])
    data_index += batch_size
    return batch

def main(_):
    
    num_pixels = 3072 #32*32*3
    
    
    trainX, trainY, testX, testY = my_data(FLAGS.data_dir)
    
    
    x = tf.placeholder(tf.float32, [None, num_pixels])

    
    y_ = tf.placeholder(tf.float32, [None, FLAGS.classes])
    
    
    y_conv, keep_prob = mymodel(x)
    
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    total_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    MY_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(MY_pred, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(FLAGS.max_steps):
            batch = generate_batch(trainX, trainY, FLAGS.batch_size)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            total_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: FLAGS.dropout_prob})
            if i % 1000 == 0:
                print('test accuracy %g' % accuracy.eval(feed_dict={
                        x: testX, y_: testY, keep_prob: 1.0}))
    
        print('Final test accuracy %g' % accuracy.eval(feed_dict={
                        x: testX, y_: testY, keep_prob: 1.0}))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='images/',
                      help='please input Directory name')
    parser.add_argument('--classes', type=int, default=3,
                      help='enter the class number')
    parser.add_argument('--batch_size', type=int, default=50,
                      help='size of batch')
    parser.add_argument('--max_steps', type=int, default=50500,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--dropout_prob', type=float, default=0.5,
                      help='ratio of dropout')
    
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
