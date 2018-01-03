import argparse
# Copyright 2017 gdyshi. All Rights Reserved.
# github: https://github.com/gdyshi
# ==============================================================================

import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

FLAGS = None

layers = [784, 270, 90, 30, 10]
TRAINING_STEPS = 20000
LR_NONE = 'none'
LR_MOMENTUM = 'monentum'
LR_NAG = 'nag'
LR_ADAGRAD = 'adagrad'
LR_RMSPROP = 'RMSprop'
LR_ADAM = 'Adam'
LR_ADADELTA = 'adadelta'

LR_DECAY_LINER = 'LINER'  # 线性下降
LR_DECAY_EXP = 'EXP'  # 指数下降
LR_DECAY_NATURAL_EXP = 'NATURAL_EXP'  # e为底的指数下降
# LR_DECAY_polynomial = 'polynomial'  # 多项式

# 0.02
# LINER、EXP、NATURAL_EXP:0.2
# ADAM:0.001
learning_rate = 0.02
batch_size = 50  # mini-batch

# OPTIMIZE_METHORD = LR_ADAM
OPTIMIZE_METHORD = LR_ADAM


def accuracy(y_pred, y_real):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_real, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    for i in range(0, len(layers) - 1):
        X = x if i == 0 else y

        node_in = layers[i]
        node_out = layers[i + 1]
        W = tf.Variable(np.random.randn(node_in, node_out).astype('float32') / (np.sqrt(node_in)))
        b = tf.Variable(np.random.randn(node_out).astype('float32'))
        z = tf.matmul(X, W) + b
        y = tf.nn.tanh(z)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(tf.norm(y_ - y, axis=1) ** 2) / 2

    global_step = tf.Variable(0, trainable=False)
    if OPTIMIZE_METHORD == LR_DECAY_EXP:
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 1000, 0.96, staircase=True)
    elif OPTIMIZE_METHORD == LR_DECAY_NATURAL_EXP:
        # decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
        decayed_learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, 1000, 0.96, staircase=True)
    elif OPTIMIZE_METHORD == LR_DECAY_LINER:
        # decayed_learning_rate = learning_rate / (1 + decay_rate * t)
        decayed_learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, 1000, 0.96, staircase=True)
    # elif OPTIMIZE_METHORD == LR_DECAY_polynomial:
    #     # global_step = min(global_step, decay_steps)
    #     # decayed_learning_rate = (learning_rate - end_learning_rate) *
    #     # (1 - global_step / decay_steps) ^ (power) +
    #     # end_learning_rate
    #     decayed_learning_rate = tf.train.polynomial_decay(learning_rate, global_step, 100, 50)
    else:
        decayed_learning_rate = learning_rate

    if OPTIMIZE_METHORD == LR_MOMENTUM:
        train_step = tf.train.MomentumOptimizer(decayed_learning_rate, 0.5).minimize(loss, global_step=global_step)
    elif OPTIMIZE_METHORD == LR_NAG:
        train_step = tf.train.MomentumOptimizer(decayed_learning_rate, 0.5, use_nesterov=True).minimize(loss,
                                                                                                        global_step=global_step)
    elif OPTIMIZE_METHORD == LR_ADAGRAD:
        train_step = tf.train.AdagradOptimizer(decayed_learning_rate).minimize(loss, global_step=global_step)
    elif OPTIMIZE_METHORD == LR_RMSPROP:
        train_step = tf.train.RMSPropOptimizer(decayed_learning_rate).minimize(loss, global_step=global_step)
    elif OPTIMIZE_METHORD == LR_ADAM:
        train_step = tf.train.AdamOptimizer(decayed_learning_rate).minimize(loss, global_step=global_step)
    elif OPTIMIZE_METHORD == LR_ADADELTA:
        train_step = tf.train.AdadeltaOptimizer(decayed_learning_rate).minimize(loss, global_step=global_step)
    else:
        train_step = tf.train.GradientDescentOptimizer(decayed_learning_rate).minimize(loss, global_step=global_step)

    acc = accuracy(y, y_)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            if i % 1000 == 0:
                valid_acc = acc.eval(feed_dict={x: mnist.validation.images,
                                                     y_: mnist.validation.labels})
                print("After %d training step(s), accuracy on validation is %g." % (i, valid_acc))
            train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
        test_acc = acc.eval(feed_dict={x: mnist.test.images,
                                         y_: mnist.test.labels})
        print("After %d training step(s), accuracy on test is %g." % (TRAINING_STEPS, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='E:\data\mnist',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
