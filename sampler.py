import tensorflow as tf
import numpy as np
from train_ops import TrainOps
from data_loader import DataLoader
from architecture import Architecture as A
from StringIO import StringIO
from tensorflow.python.lib.io import file_io
import os


class Sampler:

    def sample(self, input_images_dir, output_dir, sess):
        ops = TrainOps(sess.graph)

        # input_images dataset
        dataset = DataLoader().load_images(input_images_dir, batch_size=100)
        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()
        x_images = sess.run(batch)

        # correct num channels if needed
        channel_axis = 3
        if x_images.shape[channel_axis] < A.input_channels:
            x_images = np.repeat(x_images, A.input_channels, axis=channel_axis)

        # create feed-dict
        feed_dict = {ops.x_images_holder: x_images}

        # run generated_image
        generated_images = sess.run(ops.generated_images, feed_dict=feed_dict)
        generated_images = generated_images + 1.
        generated_images = generated_images * 128.

        # save resulting image to sample_dir
        self.save_images(generated_images, output_dir)


    def save_images(self, images, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        sess = tf.Session()
        for i in range(images.shape[0]):
            image = images[i]
            img_tensor = tf.image.encode_jpeg(image)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img_name = save_dir + '/' + 'sample_' + str(i) + '.jpeg'
            output_data = sess.run(img_tensor)
            with file_io.FileIO(img_name, 'w+') as f:
                f.write(output_data)
                f.close
