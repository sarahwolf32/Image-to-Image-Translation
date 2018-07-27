import tensorflow as tf
import numpy as np
from architecture import Architecture as A

class Generator:

    def create(self, images):
        '''
        Generates fake images based off the input images.
        Uses a "U-net" architecture, an encoder-decoder with skip connections

        @images: A set of images normalized to [-1, 1], with shape [batch_size, img_size, img_size, input_channels]
        @returns: A set of generated images in range [-1, 1], with shape [batch_size, img_size, img_size, output_channels]
        '''

        with tf.variable_scope('generator'):
            encoder_layers, encoder_layer_channels = self._encoder(images)
            decoder_layers = self._decoder(encoder_layers, encoder_layer_channels)
            output = decoder_layers[-1]
            output = tf.identity(output, name='generated_images')
            return output


    def _encoder(self, images):
        '''
        The first half of the generator.
        Reduces each image to a vector capturing its high-level features.

        @images: Tensor - A set of images normalized to [-1, 1], with shape [batch_size, img_size, img_size, input_channels]
        @returns: 
            @layers:
                A list of all encoder layers. 
                The last layer is a [batch_size, 1, 1, num_channels] encoding of the input image.
            @layer_channels
                A list of the number of channels in each layer's output activation
        '''

        layers = []
        layer_channels = []
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
                num_channels = num_channels * 2
            
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

            # leaky relu activation function
            if not last_layer:
                layer = tf.nn.leaky_relu(layer)

            # save layer
            layers.append(layer)
            layer_channels.append(num_channels)

        return (layers, layer_channels)


    def _decoder(self, encoder_layers, encoder_layer_channels):
        '''
        The second half of the generator.
        Expands the encoder's final, smallest layer of shape [batch, 1, 1, num_channels] into the output image.
        Also uses skip connections from all the encoder's layers.

        @encoder_layers: [layer] - List of all encoder layers
        @encoder_layer_channels: [Int] - List of number of channels in each encoder layer
        @returns: [layer] - A list of all decoder layers
        '''

        decoder_layers = []
        initializer = tf.truncated_normal_initializer(stddev=0.02)

        img_size = 1
        layer = encoder_layers[-1]

        while img_size < A.img_size:

            # determine if first or last layer, as they are structured a bit differently
            first_layer = (img_size == 1)
            last_layer = (img_size == A.img_size / 2)

            # update specs for layer
            img_size = img_size * 2
            skip_layer_index = len(encoder_layers) - len(decoder_layers) - 1
            num_channels = A.output_channels if last_layer else encoder_layer_channels[skip_layer_index]
            dropout = A.dropouts[len(decoder_layers)]

            # concatenate skip connection layer
            if not first_layer:
                skip_layer = encoder_layers[skip_layer_index]
                layer = tf.concat([layer, skip_layer], axis=3)

            # relu activation function
            layer = tf.nn.relu(layer)

            # deconv
            layer = tf.layers.conv2d_transpose(
                layer,
                filters = num_channels,
                kernel_size = 4,
                strides = [2,2],
                padding = "same",
                kernel_initializer=initializer)

            # droput
            layer = tf.nn.dropout(layer, keep_prob = 1 - dropout)
            
            # tanh activation function (last layer only)
            if last_layer:
                layer = tf.tanh(layer)

            # save layer
            decoder_layers.append(layer)

        return decoder_layers

            




        
            









