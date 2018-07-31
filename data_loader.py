import numpy as np
import tensorflow as tf
from architecture import Architecture as A

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

    def cast_to_channels(self, num_channels, images):
        channels_axis = 3
        is_grayscale = (images.shape[channels_axis] == 1)
        want_grayscale = (num_channels == 1)

        # reduce channels if needed
        if not is_grayscale and want_grayscale:
            images = np.mean(images, axis=channels_axis, keepdims=True)
            return images
            
        # expand channels if needed
        if is_grayscale and not want_grayscale:
            images = np.repeat(images, num_channels, axis=channels_axis)
            return images

        return images

    def split_images(self, combined_images):
        x_images = combined_images[:, :, :A.img_size, :]
        y_images = combined_images[:, :, A.img_size:, :]
        x_images = self.cast_to_channels(A.input_channels, x_images)
        y_images = self.cast_to_channels(A.output_channels, y_images)
        return (x_images, y_images)


    # HELPERS

    def _shift_and_scale(self, X):
        '''Scales image with values in range [0, 255] to range [-1, 1]'''
        X = tf.cast(X, tf.float32)
        X = X / tf.constant(128., dtype=tf.float32)
        X = X - tf.constant(1., dtype=tf.float32)
        return X








    