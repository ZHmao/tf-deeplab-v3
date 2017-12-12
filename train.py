#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import os
import time
import random

import tensorflow as tf
import cv2

from net_help import load_model, save_model, variable_summaries
from net_small import DeepLabResNetModel
from utils import decode_labels, prepare_labels, inv_preprocess
from defaults import *


def read_data():
    h, w = INPUT_SIZE
    train_dir = TRAIN_DATA_DIR
    val_dir = VAL_DATA_DIR
    results = []
    for tmp_dir in [train_dir, val_dir]:
        img_dir = os.path.join(tmp_dir, 'images')
        mask_dir = os.path.join(tmp_dir, 'labels')

        tmp_data = []
        for file_name in os.listdir(img_dir):
            p_fn = file_name[:-4]
            img = cv2.imread(os.path.join(img_dir, file_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            img  = img.astype('float32') - IMG_MEAN

            mask = cv2.imread(os.path.join(mask_dir, p_fn+'.png'),
                              cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
            mask = np.divide(mask, 255)
            mask = np.expand_dims(mask, axis=2)
            tmp_data.append((img, mask))
        results.append(tmp_data)

    train_data = results[0]
    val_data = results[1]
    return (train_data, val_data)



def main():
    # Create model and start training
    h, w = INPUT_SIZE

    X = tf.placeholder(tf.float32, shape=[None, h, w, 3], name='X')
    Y = tf.placeholder(tf.uint8, shape=[None, h, w, 1], name='Y')
    is_training = tf.placeholder(tf.bool, name='is_training')

    net = DeepLabResNetModel(X, is_training, NUM_CLASSES, ATROUS_BLOCKS)

    raw_output = net.output

    # Trainable Variables
    # restore_vars = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name]
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name]
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name]

    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_prediction = tf.reshape(raw_output, [-1, NUM_CLASSES])
    label_proc = prepare_labels(Y, tf.stack(raw_output.get_shape()[1:3]), num_classes=NUM_CLASSES, one_hot=False)
    raw_gt = tf.reshape(label_proc, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, NUM_CLASSES - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)

    # Pixel-wise Softmax Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    l2_losses = [WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
    variable_summaries(reduced_loss, name='loss')
    variable_summaries(loss, name='loss_origin')

    # Processed predictions: for visualization
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(X)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Define loss and optimization parameters
    base_lr = tf.constant(LEARNING_RATE, tf.float64)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    increment_step = tf.assign(global_step, global_step + 1)
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - global_step / NUM_STEPS), POWER))
    learning_rate = tf.maximum(learning_rate, 8e-7)

    opt_conv = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 5.0, MOMENTUM)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 10.0, MOMENTUM)

    grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
    grads_conv = grads[:len(conv_trainable)]
    grads_fc_w = grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.group(increment_step, train_op_conv, train_op_fc_w, train_op_fc_b)

    # initial_learning_rate = 1e-2
    # learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 300, 0.96)
    # adam = tf.train.AdamOptimizer(learning_rate).minimize(reduced_loss, global_step=global_step)

    # Image Summary
    images_summary = tf.py_func(inv_preprocess, [X, SAVE_NUM_IMAGES, IMG_MEAN], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, SAVE_NUM_IMAGES, NUM_CLASSES], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [Y, SAVE_NUM_IMAGES, NUM_CLASSES], tf.uint8)

    image_summaries = [images_summary, preds_summary, labels_summary]
    image_summary = tf.summary.image('images',
                                     tf.concat(axis=2, values=image_summaries),
                                     max_outputs=SAVE_NUM_IMAGES)

    # Variable Summary
    variable_summaries(fc_w_trainable, 'fc_w')
    variable_summaries(fc_b_trainable, 'fc_b')
    variable_summaries(learning_rate, 'learning_rate')
    # variable_summaries(net.weights, 'aconv_w')
    # variable_summaries(net.biases, 'aconv_b')

    total_summary = tf.summary.merge_all()

    tb_train_dir = os.path.join(SNAPSHOT_DIR, 'train')
    tb_val_dir = os.path.join(SNAPSHOT_DIR, 'verify')
    summary_writer = tf.summary.FileWriter(tb_train_dir,
                                           graph=tf.get_default_graph())
    verify_writer = tf.summary.FileWriter(tb_val_dir)

    train_data, val_data = read_data()
    train_data_count = len(train_data)
    batch_size = BATCH_SIZE

    print('\n batch size: %s, total step: %s, train data num: %s \n' %
          (batch_size, NUM_STEPS, train_data_count))
    # Set up session
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # save_var = tf.trainable_variables()
        # bn_moving_vars = [v for v in tf.global_variables() \
        #                   if
        #                   '/moving_mean' in v.name or '/moving_variance' in v.name]
        # save_var += bn_moving_vars
        saver = tf.train.Saver(max_to_keep=3)
        if SNAPSHOT_DIR is not None and os.path.exists(SNAPSHOT_DIR):
            loader = tf.train.Saver()
            load_model(loader, sess, SNAPSHOT_DIR)

        start_index = 0
        for step in range(NUM_STEPS):
            # start_time = time.time()

            if start_index + batch_size > train_data_count:
                start_index = 0
                random.shuffle(train_data)
            _train_data = train_data[start_index: start_index + batch_size]
            start_index += batch_size
            after_processing_data = []
            start_time = time.time()
            for x_img, y_data in _train_data:
                # tmp_img = augmentation(x_img)
                # x_data = np.asarray(tmp_img).astype('float32') / 255.0
                after_processing_data.append((x_img, y_data))
            vec = [d[0] for d in after_processing_data]
            vec2 = [d[1] for d in after_processing_data]
            # data_process_time = time.time() - start_time
            if step % SAVE_SUMMARY_EVERY == 0:
                feed = [reduced_loss, pred,
                        total_summary, global_step, train_op]
                loss_value, preds, summary, total_steps, _ = \
                    sess.run(feed, feed_dict={X: vec, Y: vec2, is_training: True})
                summary_writer.add_summary(summary, total_steps)
            else:
                feed = [reduced_loss, global_step, train_op]
                loss_value, total_steps, _ = sess.run(feed, feed_dict={X: vec, Y: vec2, is_training: True})
            if step % SAVE_MODEL_EVERY == 0:
                save_model(saver, sess, SNAPSHOT_DIR, global_step)

            duration = time.time() - start_time
            results = 'global step: {:d}, step: {:d} \t loss = {:.3f}, ({:.3f} secs)'\
                .format(total_steps, step, loss_value, duration)
            if step % WRITE_EVERY == 0:
                with open(WRITE_FILE, 'a') as f:
                    f.write(results + '\n')
            print(results)

            if step % VAL_EVERY == 0:
                random.shuffle(val_data)
                vec = [d[0] for d in val_data[:batch_size]]
                vec2 = [d[1] for d in val_data[:batch_size]]
                feed = [reduced_loss, pred,
                        total_summary, global_step]
                loss_value, preds, summary, total_steps = \
                    sess.run(feed, feed_dict={X: vec, Y: vec2, is_training: False})
                verify_writer.add_summary(summary, total_steps)
                print('[Verify]', step, loss_value)


if __name__ == '__main__':
    main()
