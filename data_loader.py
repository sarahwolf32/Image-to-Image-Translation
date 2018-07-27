import numpy as np
import tensorflow as tf

class DataLoader:

    ''' 
    Creates a dataset of images, and scales them to [-1, 1].
    Implements batch size.
    Assumes the images are jpegs.
    '''

    def __init__(self, images_dir, batch_size):
        self.images_dir = images_dir
        self.batch_size = batch_size

    def _load_images(self):

        # make a Dataset of all filenames
        file_pattern = self.images_dir + "/*.jpeg"
        filename_dataset = tf.data.Dataset.list_files(file_pattern)
        
        # convert dataset to image tensors
        dataset = filename_dataset.map(lambda x: tf.image.decode_jpeg(tf.read_file(x)))

        # TODO: scale image tensor values to [-1, 1]

        # mini-batches
        dataset = dataset.batch(self.batch_size)

        return dataset






    