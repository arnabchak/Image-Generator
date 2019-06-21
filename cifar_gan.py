# -*- coding: utf-8 -*-
"""
Created on Sun Mar 3 17:42:47 2019

@author: Arnab Chakravarty
"""
import keras
import gan_model
import numpy as np
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image
from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from sklearn.utils import shuffle
import os
import time

rows = 100
cols = 100
channels = 3
dims = 25
img_input = Input(shape=(rows, cols, channels))
discriminator, disc_out = gan_model.get_discriminator(img_input)
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
discriminator.trainable = False
noise_input = Input(shape=(100,))
generator, gen_out = gan_model.get_generator(noise_input, dims, channels)

gan_input = Input(shape=(100,))
x = generator(gan_input)
gan_out = discriminator(x)
gan = Model(gan_input, gan_out)
#gan.load_weights('GAN-96.h5')
gan.summary()
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

def save_imgs(epoch):
      noise = gan_model.generate_noise(9, 100)
      gen_imgs = generator.predict(noise)
      img = image.array_to_img(gen_imgs[2])
      img.save(os.path.join(save_dir, 'generated_img_' + str(epoch) + '.png'))

def read_image(path):
    img = load_img(path, target_size=(100,100))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def file_length(path):
    i = 0
    for file in listdir(path):
        i += 1
    return i

BATCH_SIZE = 20
N_EPOCHS = 500
DISC_TRAIN_RATIO = 3

save_dir = 'C:/Users/Arnab Chakravarty/Desktop/project/img_gen/Results'
path = 'Tomato'
length = file_length(path)
X_train = np.zeros((length, 100, 100, 3))
j = 0
for name in listdir(path):
    filename = path + '/' + name
    img_arr = read_image(filename)
    X_train[j, :] = img_arr
    j += 1
np.random.shuffle(X_train)

for i in range(5):
    img = image.array_to_img(X_train[i])
    img.save(os.path.join(save_dir, 'actual_img_' + str(i) + '.png'))
X_train = (X_train - 127.5) / 127.5 
num_batches = int(X_train.shape[0]/BATCH_SIZE)

for epoch in range(N_EPOCHS):
      cum_d_loss = 0
      cum_g_loss = 0
      start = time.time()
      for batch_idx in range(num_batches):
        images = X_train[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
        noise_data = gan_model.generate_noise(BATCH_SIZE, 100)
        generated_images = generator.predict(noise_data)
    
        noise_prop = 0.05 
        true_labels = np.zeros((BATCH_SIZE, 1)) + np.random.uniform(low=0.0, high=0.1, size=(BATCH_SIZE, 1))
        flipped_idx = np.random.choice(np.arange(len(true_labels)), size=int(noise_prop*len(true_labels)))
        true_labels[flipped_idx] = 1 - true_labels[flipped_idx]
        
        gene_labels = np.ones((BATCH_SIZE, 1)) - np.random.uniform(low=0.0, high=0.1, size=(BATCH_SIZE, 1))
        flipped_idx = np.random.choice(np.arange(len(gene_labels)), size=int(noise_prop*len(gene_labels)))
        gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]
        
        for _ in range(DISC_TRAIN_RATIO):
            d_loss_true = discriminator.train_on_batch(images, true_labels)
            d_loss_gene = discriminator.train_on_batch(generated_images, gene_labels)
            d_loss = 0.5 * np.add(d_loss_true, d_loss_gene)
            cum_d_loss += d_loss
        cum_d_loss /= DISC_TRAIN_RATIO
        noise_data = gan_model.generate_noise(BATCH_SIZE, 100)
        g_loss = gan.train_on_batch(noise_data, np.zeros((BATCH_SIZE, 1)))
        cum_g_loss += g_loss
      finish = time.time()
      X_train = shuffle(X_train,random_state=0)
      print('Epoch: {}, Generator Loss: {}, Discriminator Loss: {}, time_taken: {}'.format(epoch+1, cum_g_loss/num_batches, cum_d_loss/num_batches, finish-start))
      gan.save_weights(path +'-GAN-{}.h5'.format(epoch+1))
      save_imgs(epoch)
      #img_layer_11(epoch)
    
            
            
            