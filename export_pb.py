#!/usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import print_function
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

from defaults import *
from net_small import DeepLabResNetModel


def save_graph_with_weight(sess, output_node, out_file_path):
    input_graph_def = sess.graph.as_graph_def()
    for node in input_graph_def.node:
        print(node.name)
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node)
    with gfile.GFile(out_file_path, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def ckp2pb(model_path, out_pb_path):
    h, w = INPUT_SIZE

    X = tf.placeholder(tf.float32, shape=[None, h, w, 3], name='X')
    Y = tf.placeholder(tf.uint8, shape=[None, h, w, 1], name='Y')
    is_training = tf.placeholder(tf.bool, name='is_training')

    net = DeepLabResNetModel(X, is_training, NUM_CLASSES, ATROUS_BLOCKS)

    raw_output = net.output
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(X)[1:3, ])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    save_graph_with_weight(sess, ['ExpandDims'], out_pb_path)


def main():
    model_dir = './snapshots/recent/model.ckpt-60004'
    out_pb_path = '/data/pb/traffic_line_deeplab_resnet_cut05_271_loss_0_0_31.pb'
    ckp2pb(model_dir, out_pb_path)


if __name__ == '__main__':
    main()
