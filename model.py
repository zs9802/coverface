import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
num_keep_radio = 0.7


def P_Net(inputs, label=None, bbox_target=None, landmark_target=None):
    with tf.variable_scope('PNet'):
        with slim.arg_scope([slim.conv2d], activation_fn=prelu,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            padding='VALID'):
            net = slim.conv2d(inputs, 10, 3, scope='conv1')
            net = slim.max_pool2d(
                net, kernel_size=[2, 2], stride=2, padding='SAME', scope='pool1')
            net = slim.conv2d(net, 16, 3, scope='conv2')
            net = slim.conv2d(net, 32, 3, scope='conv3')
            conv4_1 = slim.conv2d(
                net, 2, 1, activation_fn=tf.nn.softmax, scope='conv4_1')
            bbox_pred = slim.conv2d(
                net, 4, 1, activation_fn=None, scope='conv4_2')
            landmark_pred = slim.conv2d(
                net, 10, 1, activation_fn=None, scope='conv4_3')

            cls_pro_test = tf.squeeze(conv4_1, axis=0)
            bbox_pred_test = tf.squeeze(bbox_pred, axis=0)
            landmark_pred_test = tf.squeeze(landmark_pred, axis=0)
            return cls_pro_test, bbox_pred_test, landmark_pred_test


def R_Net(inputs, label=None, bbox_target=None, landmark_target=None):
    with tf.variable_scope('RNet'):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=prelu,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            padding='VALID'):
            net = slim.conv2d(inputs, 28, 3, scope='conv1')
            net = slim.max_pool2d(
                net, kernel_size=[3, 3], stride=2, padding='SAME', scope='pool1')
            net = slim.conv2d(net, 48, 3, scope='conv2')
            net = slim.max_pool2d(
                net, kernel_size=[3, 3], stride=2, scope='pool2')
            net = slim.conv2d(net, 64, 2, scope='conv3')
            fc_flatten = slim.flatten(net)
            fc1 = slim.fully_connected(
                fc_flatten, num_outputs=128, scope='fc1')

            cls_prob = slim.fully_connected(
                fc1, num_outputs=2, activation_fn=tf.nn.softmax, scope='cls_fc')
            bbox_pred = slim.fully_connected(
                fc1, num_outputs=4, activation_fn=None, scope='bbox_fc')
            landmark_pred = slim.fully_connected(
                fc1, num_outputs=10, activation_fn=None, scope='landmark_fc')

            return cls_prob, bbox_pred, landmark_pred


def O_Net(inputs, label=None, bbox_target=None, landmark_target=None):
    with tf.variable_scope('ONet'):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=prelu,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            padding='VALID'):
            net = slim.conv2d(inputs, 32, 3, scope='conv1')
            net = slim.max_pool2d(
                net, kernel_size=[3, 3], stride=2, padding='SAME', scope='pool1')
            net = slim.conv2d(net, 64, 3, scope='conv2')
            net = slim.max_pool2d(
                net, kernel_size=[3, 3], stride=2, scope='pool2')
            net = slim.conv2d(net, 64, 3, scope='conv3')
            net = slim.max_pool2d(
                net, kernel_size=[2, 2], stride=2, padding='SAME', scope='pool3')
            net = slim.conv2d(net, 128, 2, scope='conv4')
            fc_flatten = slim.flatten(net)
            fc1 = slim.fully_connected(
                fc_flatten, num_outputs=256, scope='fc1')
            cls_prob = slim.fully_connected(
                fc1, num_outputs=2, activation_fn=tf.nn.softmax, scope='cls_fc')
            bbox_pred = slim.fully_connected(
                fc1, num_outputs=4, activation_fn=None, scope='bbox_fc')
            landmark_pred = slim.fully_connected(
                fc1, num_outputs=10, activation_fn=None, scope='landmark_fc')

            return cls_prob, bbox_pred, landmark_pred


def prelu(inputs):
    alphas = tf.get_variable('alphas', shape=inputs.get_shape()[-1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas*(inputs-abs(inputs))*0.5
    return pos+neg
