import tensorflow as tf
import numpy as np
from data_loader import DataLoader
from model import Model



def train():

    num_epochs = 2
    images_dir = 'flower-sketches'
    batch_size = 10

    # get trainers
    train_d, train_g, loss_d, loss_g, generated_images = Model().trainers()

    # get dataset
    loader = DataLoader(images_dir, batch_size)
    dataset = loader.load_images()

    with tf.Session() as sess:

        # loop through epochs
        for epoch in num_epochs:

            print("epoch: " + str(epoch))
            iterator = dataset.make_one_shot_iterator()     

            # loop through batches
            while True:
                try: 
                    batch = sess.run(iterator.get_next())
                    #....
                    
                except tf.errors.OutOfRangeError:
                    break





