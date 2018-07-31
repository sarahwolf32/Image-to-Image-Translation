import tensorflow as tf
import numpy as np
from architecture import Architecture as A

class Discriminator():

    def score_patches(self, input_images, output_images, reuse=False):
        '''
        A PatchGAN discriminator. 
        Uses a convolutional net to create a 30x30 array of outputs X, where:
            - each entry Xij signifies if patch ij in the image is real or fake
            - each entry Xij comes from a 70x70 region of the input 
        @input_images: A set of image tensors, of shape [batch_size, img_size, img_size, input_channels] 
        @output_images: A set of image tensors, of shape [batch_size, img_size, img_size, output_channels]
        @returns: [batch_size, 30, 30, 1] with all values in range [0,1]
        '''

        with tf.variable_scope('discriminator', reuse=reuse):

            # concatenate the channels of the generated and ground truth output images
            layer = tf.concat([input_images, output_images], axis=3)

            # prepare to create layers
            discriminator_strides = [2, 2, 2, 1, 1]
            num_channels = A.discriminator_filter_size
            max_channels = num_channels * A.max_channel_multiplier
            initializer = tf.truncated_normal_initializer(stddev=0.02)

            # create layers
            for i in range(len(discriminator_strides)):

                first_layer = (i == 0)
                last_layer = (i == len(discriminator_strides) - 1)

                # convolution
                s = discriminator_strides[i]
                layer = tf.pad(layer, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                layer = tf.layers.conv2d(
                    layer,
                    filters = 1 if last_layer else num_channels,
                    kernel_size = 4,
                    padding = "valid",
                    strides = [s, s],
                    kernel_initializer = initializer
                )
                
                # update num filters
                if num_channels < max_channels:
                    num_channels = num_channels * 2

                # batchnorm
                if not first_layer or last_layer:
                    layer = tf.layers.batch_normalization(layer)

                # activation function
                if last_layer:
                    layer = tf.sigmoid(layer)
                else:
                    layer = tf.nn.leaky_relu(layer)

            return layer

            



        
