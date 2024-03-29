import tensorflow as tf
from tensorlayer.layers import (Input, Conv2d, BatchNorm2d, Elementwise, SubpixelConv2d)
from tensorlayer.models import Model


class SRGAN:

    def get_G(self, input_shape):
        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)

        nin = Input(input_shape)
        n = Conv2d(64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init)(nin)
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
            nn = BatchNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
            nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)
            nn = BatchNorm2d(gamma_init=g_init)(nn)
            nn = Elementwise(tf.add)([n, nn])
            n = nn

        n = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
        n = BatchNorm2d(gamma_init=g_init)(n)
        n = Elementwise(tf.add)([n, temp])
        # B residual blacks end

        n = Conv2d(256, (3, 3), (1, 1), padding='SAME', W_init=w_init)(n)
        n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)

        n = Conv2d(256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init)(n)
        n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)

        nn = Conv2d(3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init)(n)
        G = Model(inputs=nin, outputs=nn)
        return G
