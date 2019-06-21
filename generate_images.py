# -*- coding: utf-8 -*-
"""
Created on Sat Mar 9 15:33:21 2019

@author: Arnab Chakravarty
"""

import gan_model
from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image
import os
 

channels = 3
save_dir = 'C:/Users/Arnab Chakravarty/Desktop/project/img_gen/Results'
fruit = input("which images do you want to generate?  ")
size = input("what will be the size of image?  ")
fruit = fruit.lower()
copy = input("how many images do you want?  ")
copy = int(copy)
size = int(size)
rows = size
cols = size
dims = int(size/4)
img_input = Input(shape=(rows, cols, channels))
discriminator, disc_out = gan_model.get_discriminator(img_input)
noise_input = Input(shape=(100,))
generator, gen_out = gan_model.get_generator(noise_input, dims, channels)
gan_input = Input(shape=(100,))
x = generator(gan_input)
gan_out = discriminator(x)
gan = Model(gan_input, gan_out)
gan.load_weights(fruit + '-GAN.h5')
gan.trainable = False
gan.summary()

def save_imgs(epoch):
      for i in range(epoch):
          noise = gan_model.generate_noise(1, 100)
          gen_imgs = generator.predict(noise)
          img = image.array_to_img(gen_imgs[0])
          img.save(os.path.join(save_dir, 'generated_Img_' + str(i) + '.png'))

save_imgs(copy)