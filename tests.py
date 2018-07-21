import unittest
import numpy as np
import tensorflow as tf
from architecture import Architecture as A
from generator import Generator

class Tests(unittest.TestCase):

    def test_architecture(self):
        self.assertTrue(np.log2(A.img_size).is_integer(), msg='img_size must be a power of 2')
        self.assertTrue(np.log2(A.max_channel_multiplier).is_integer(), msg='max_channel_multiplier must be a power of 2')
        self.assertEqual(np.log2(A.img_size), len(A.dropouts), msg='dropouts list must be correct length')
        self.assertEqual(A.dropouts[-1], 0.0, msg='last generator decoder layer should not have dropout')

    def test_generator(self):
        G = Generator()

        # create fake tensor of shape [batch_size, img_size, img_size, input_channels]
        batch_size = 128
        input_shape = [batch_size, A.img_size, A.img_size, A.input_channels]
        images = tf.random_uniform(input_shape, minval = -1, maxval = 1)
        
        # get output from G
        output = G.generate(images)

        # confirm output shape is as expected
        expected_output_shape = [batch_size, A.img_size, A.img_size, A.output_channels]
        self.assertEqual(output.shape, expected_output_shape, msg='generator should output tensor of size [batch_size, img_size, img_size, output_channels]')



if __name__ == '__main__':
    unittest.main()