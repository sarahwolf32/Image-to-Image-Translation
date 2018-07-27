import tensorflow as tf
import numpy as np
from data_loader import DataLoader
from model import Model
from train_ops import TrainOps
from architecture import Architecture as A
import argparse
from logger import Logger


def train(config):

    # initialize model
    Model()

    # get dataset
    loader = DataLoader(config)
    dataset = loader.load_images()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # get train ops
        sess.run(init)
        ops = TrainOps(sess.graph)
        logger = Logger(config, sess, ops)
        epoch = sess.run(ops.epoch)

        # loop through epochs
        while epoch < config.num_epochs:
            iterator = dataset.make_one_shot_iterator()     

            # loop through batches
            while True:
                try: 

                    # get mini-batch 
                    batch = sess.run(iterator.get_next())
                    x_images = batch[:, :, :A.img_size, :]
                    y_images = batch[:, :, A.img_size:, :]
                    
                    # train
                    feed_dict = {ops.x_images_holder: x_images, ops.y_images_holder: y_images}
                    sess.run([ops.train_g, ops.train_d], feed_dict=feed_dict)
                    logger.log(feed_dict)
                    sess.run(tf.assign_add(ops.global_step, 1))
                    
                except tf.errors.OutOfRangeError:
                    break

            # increment epoch
            sess.run(tf.assign_add(ops.epoch, 1))
            epoch = sess.run(ops.epoch)




# MAIN
if __name__=='__main__':

    # unwrap config
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='flower-sketches')
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--summary-dir', default='summary')
    config = parser.parse_args()

    # train
    train(config)





