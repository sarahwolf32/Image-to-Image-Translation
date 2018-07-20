import tensorflow as tf
import numpy as np
from architecture import Architecture as A

class Generator:

    def generate(self, images):
        '''
        Generates fake images based off the input images.
        Uses a "U-net" architecture, an encoder-decoder with skip connections

        @images: A set of images normalized to [-1, 1], with shape [batch_size, img_size, img_size, input_channels]
        @returns: A set of generated images in range [-1, 1], with shape [batch_size, img_size, img_size, output_channels]
        '''

        with tf.variable_scope('generator'):
            layers = self.encoder(images)
            #....


    def encoder(self, images):
        '''
        The first half of the generator.
        Reduces each image to a vector capturing its high-level features.

        @images: A set of images normalized to [-1, 1], with shape [batch_size, img_size, img_size, input_channels]
        @returns: 
            A list of all encoder layers. 
            The last layer is a [batch_size, 1, 1, num_channels] encoding of the input image.
        '''

        layers = []
        initializer = tf.truncated_normal_initializer(stddev=0.02)

        # layer size: [batch, img_size, img_size, num_channels]
        img_size = A.img_size
        max_channels = A.max_channel_multiplier * A.input_channels
        num_channels = A.input_channels
        layer = images

        while img_size > 1:

            # determine if first or last layer, as they are structured a bit differently
            first_layer = (img_size == A.img_size)
            last_layer = (img_size == 2)

            # update dimensions for layer
            img_size = img_size / 2
            if num_channels < max_channels:
                num_channels * 2

            # conv
            layer = tf.layers.conv2d(
                layer, 
                filters = num_channels,
                kernel_size = 4,
                strides = [2,2],
                padding = 'same',
                kernel_initializer = initializer)

            # batch norm
            if not first_layer:
                layer = tf.layers.batch_normalization(layer)

            # leaky relu
            if not last_layer:
                layer = tf.nn.leaky_relu(layer)

            # save layer
            layers.append(layer)

        return layers


            

            
            
            

            

        # decoder
            # if first decoder layer:
                # input = previous_layer
            # else:
                # input = previous_layer + skip_connection_layer, concatenated on channels axis
            # All but the last decoder layer consists of:
                # 1. get input
                # 2. relu
                # 3. deconv
                # 4. dropout
            # The last layer consists of:
                # 1. get input
                # 2. relu
                # 3. deconv
                # 4. tanh
            # Img_size should double each layer, until output size is reached
            # num_channels output should mirror encoder
            # return last layer
            









