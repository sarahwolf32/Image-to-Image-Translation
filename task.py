import tensorflow as tf
import numpy as np
from data_loader import DataLoader
from model import Model
from train_ops import TrainOps
from architecture import Architecture as A



def train():

    num_epochs = 2
    images_dir = 'flower-sketches'
    batch_size = 10

    # initialize model
    Model()

    # get dataset
    loader = DataLoader(images_dir, batch_size)
    dataset = loader.load_images()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # get train ops
        sess.run(init)
        ops = TrainOps(sess.graph)

        # loop through epochs
        for epoch in range(num_epochs):

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

train()





