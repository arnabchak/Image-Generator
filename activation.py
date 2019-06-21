# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 23:58:29 2019

@author: Arnab Chakravarty
"""

import gan_model
from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image
import os
import numpy as np 
import matplotlib.pyplot as plt
channels = 3
save_dir = 'C:/Users/Arnab Chakravarty/Desktop/project/img_gen/Results'
fruit = input("which images do you want to generate?  ")
size = input("what will be the size of image?  ")
fruit = fruit.lower()
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

layer_outputs = [layer.output for layer in discriminator.layers[1:10]]
activation_model = Model(inputs=discriminator.get_input_at(0), outputs=layer_outputs)
activation_model.summary()

noise = gan_model.generate_noise(1, 100)
gen_imgs = generator.predict(noise)
img = image.array_to_img(gen_imgs[0])
img = np.expand_dims(img, axis=0)

activations = activation_model.predict(img)
#print(len(activations))
first_layer_activation = activations[8]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 8], cmap='viridis')
plt.show()