import unittest
import numpy as np
from architecture import Architecture as A

class Tests(unittest.TestCase):

    def test_architecture(self):
        self.assertTrue(np.log2(A.img_size).is_integer(), msg='img_size must be a power of 2')
        self.assertTrue(np.log2(A.max_channel_multiplier).is_integer(), msg='max_channel_multiplier must be a power of 2')
        self.assertEqual(np.log2(A.img_size), len(A.dropouts), msg='dropouts list must be correct length')
        self.assertEqual(A.dropouts[-1], 0.0, msg='last generator decoder layer should not have dropout')





if __name__ == '__main__':
    unittest.main()