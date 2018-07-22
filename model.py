import tensorflow as tf
import numpy as np
from architecture import Architecture as A

class Model:
 
    def loss(self, Dx, Dg, y, generated_y):
        '''
        Dx: Probabilities assigned by D to the real image patches, [batch_size, 30, 30, 1]
        Dg: Probabilities assigned by D to the fake image patches, [batch_size, 30, 30, 1]
        Returns: (Discriminator Loss, Generator Loss) 
        '''

        # small epsilon value to prevent nan errors
        e = 1e-12

        # discriminator loss
        loss_d = tf.reduce_mean(-tf.log(Dx + e) - tf.log(1. - Dg + e))

        # generator loss
        loss_gan = tf.reduce_mean(-tf.log(Dg + e))
        loss_L1 = tf.reduce_mean(tf.abs(y - generated_y))
        loss_g = A.weight_gan_loss * loss_gan + A.weight_L1_loss * loss_L1

        return (loss_d, loss_g)

        