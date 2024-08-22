from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import matplotlib.pyplot as plt

import sys

import numpy as np

class MY_GAN():
    def __init__(self):
        self.rows = 28
        self.cols = 28
        self.channels = 1
        self.picture_shape = (self.rows, self.cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminative_model()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generate_model()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminative_model(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.picture_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.picture_shape)
        validity = model(img)

        return Model(img, validity)

    def build_generate_model(self):
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.picture_shape), activation='tanh'))
        model.add(Reshape(self.picture_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def gan_train(self, epochs, batch_size=128, simple_interval=50):
        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            # 产生随机真图片
            inx = np.random.randint(0, X_train.shape[0], batch_size)
            suiji_img = X_train[inx]
            # 生成噪声，产出假的的图片
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            generate_img = self.generator.predict(noise)
            # 这里使用train_on_batch api 非test_on_batch
            zhen_loss = self.discriminator.train_on_batch(suiji_img, valid)
            fack_loss = self.discriminator.train_on_batch(generate_img, fake)
            loss_mean = 0.5 * np.add(zhen_loss, fack_loss)

            # 利用新噪声，产生新的假图片
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gan_loss = self.combined.train_on_batch(noise, valid)
            # 打印进度图
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, loss_mean[0], 100 * loss_mean[1], gan_loss))
            # 每循环simple_interval次，保存一次模型
            if epoch % simple_interval == 0:
                self.simple_img(epoch)

    def simple_img(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("./images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    my_gan = MY_GAN()
    my_gan.gan_train(epochs=2000, batch_size=32, simple_interval=200)






