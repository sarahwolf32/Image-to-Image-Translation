import tensorflow as tf
import numpy as np
from architecture import Architecture as A
from generator import Generator 
from discriminator import Discriminator

class Model:

    def __init__(self):
        self._create()

    def _create(self):

        # placeholders for training data
        x_images_holder = tf.placeholder(tf.float32, shape=[None, A.img_size, A.img_size, A.input_channels])
        y_images_holder = tf.placeholder(tf.float32, shape=[None, A.img_size, A.img_size, A.output_channels])

        # forward pass
        G = Generator()
        D = Discriminator()
        generated_images = G.create(x_images_holder)
        Dx = D.score_patches(x_images_holder, y_images_holder)
        Dg = D.score_patches(x_images_holder, generated_images, reuse=True)
        
        # compute losses
        loss_d, loss_g = self._loss(Dx, Dg, y_images_holder, generated_images)

        # optimizers
        optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

        # backprop
        g_vars = tf.trainable_variables(scope='generator')
        d_vars = tf.trainable_variables(scope='discriminator')
        train_g = optimizer_g.minimize(loss_g, var_list=g_vars, name='train_g')
        train_d = optimizer_d.minimize(loss_d, var_list=d_vars, name='train_d')

 
    def _loss(self, Dx, Dg, y, generated_y):
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

        # name both tensors
        loss_d = tf.identity(loss_d, name='loss_d')
        loss_g = tf.identity(loss_g, name='loss_g')

        return (loss_d, loss_g)

    

        