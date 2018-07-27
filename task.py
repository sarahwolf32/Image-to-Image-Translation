import tensorflow as tf
import numpy as np
from data_loader import DataLoader
from model import Model
from train_ops import TrainOps
from architecture import Architecture as A
import argparse

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

        # loop through epochs
        for epoch in range(config.num_epochs):

            print("epoch: " + str(epoch))
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

                    # log
                    loss_g, loss_d = sess.run([ops.loss_g, ops.loss_d], feed_dict=feed_dict)
                    print("loss_g: " + str(loss_g))
                    print("loss_d: " + str(loss_d))
                    
                except tf.errors.OutOfRangeError:
                    break


# MAIN
if __name__=='__main__':

    # unwrap config
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='flower-sketches')
    parser.add_argument('--num-epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=10)
    config = parser.parse_args()

    # train
    train(config)





