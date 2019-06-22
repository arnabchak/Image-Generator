# Image-Generator
A neural Image Generator using DCGAN (Deep Convolutional Generative Adversarial network) that can generate images comparable to the real images, on which the model is trained.

DATASET : CIFAR-10, Flowers, Fruits-360, LSUN-Bedroom

# all the code is  written in Keras and can be trained both on CPU & GPU ( GPU is always preferable ).

gan_model.py -> to define the GAN model

cifar_gan.py -> to train the model & save the best model.

generate_image.py -> to generate the required images from the saved models, without any further training.

activation.py -> to visualise the intermediate activations.
