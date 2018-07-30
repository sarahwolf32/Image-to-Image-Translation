import numpy as np
import tensorflow as tf

class DataLoader:

    ''' 
    Creates a dataset of images, and scales them to [-1, 1].
    Implements batch size.
    Assumes the images are jpegs.
    '''

    def load_images(self, images_dir, batch_size=None):

        # make a Dataset of all filenames
        file_pattern = images_dir + "/*.jpeg"
        filename_dataset = tf.data.Dataset.list_files(file_pattern)
        
        # convert dataset to image tensors
        dataset = filename_dataset.map(lambda x: tf.image.decode_jpeg(tf.read_file(x)))

        # scale image tensor values to [-1, 1]
        dataset = dataset.map(lambda x: self._shift_and_scale(x))

        # mini-batches
        if batch_size:
            dataset = dataset.batch(batch_size)

        return dataset


    # HELPERS

    def _shift_and_scale(self, X):
        '''Scales image with values in range [0, 255] to range [-1, 1]'''
        X = tf.cast(X, tf.float32)
        X = X / tf.constant(128., dtype=tf.float32)
        X = X - tf.constant(1., dtype=tf.float32)
        return X








    