
from keras.datasets.cifar10 import load_data
from numpy.random import randn
from numpy.random import randint
import numpy as np
import tensorflow as tf

''' 
This is loading and getting values between -1 and 1 as the generator has a TANH function
'''
def load_real_samples():
	(trainX, _), (_, _) = load_data()
	# cConvert to float and scale.
	X = trainX.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5  #Generator uses tanh activation so rescale 
                            #original images to -1 to 1 to match the output of generator.
	return X



def generate_real_samples(dataset, n_samples):
	# choose random images
	ix = randint(0, dataset.shape[0], n_samples)
	# select the random images and assign it to X
	X = dataset[ix]
	# generate class labels and assign to y
	y = np.ones((n_samples, 1)) ##Label=1 indicating they are real
	return X, y



''' 
Creating the latent vectors
'''
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


''' 
Creating the  fakes images.
And setting the label as 1 (real image)
'''
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict using generator to generate fake samples. 
	X = generator.predict(x_input)
	# Class labels will be 0 as these samples are fake. 
	y = np.zeros((n_samples, 1))  #Label=0 indicating they are fake
	return X, y


''' 
This is the function for the training.
I takes half batch of real and half batch of fake for training the discriminator
then train  only the generator with the combined model (where the discriminator is set to not trainable).
Basically, here the combined model gets a batch of fake images with the label 1 and tries to minimize its loss that would be creating the best possible images 
that the discriminator would see as real. 
'''
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)  #the discriminator model is updated for a half batch of real samples 
                            #and a half batch of fake samples, combined a single batch. 
    # manually enumerate epochs and bacthes. 
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            
            # Train the discriminator on real and fake images, separately (half batch each)
            #Research showed that separate training is more effective. 
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            ##train_on_batch allows you to update weights based on a collection 
            #of samples you provide
            #Let us just capture loss and ignore accuracy value (2nd output below)
            d_loss_real, _ = d_model.train_on_batch(X_real, y_real) 


            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss_fake, _ = d_model.train_on_batch(X_fake, y_fake)

            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
                
            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            #This is where the generator is trying to trick discriminator into believing
            #the generated image is true (hence value of 1 for y)			
            y_gan = np.ones((n_batch, 1))

            # Generator is part of combined model where it got directly linked with the discriminator
            # Train the generator with latent_dim as x and 1 as y. 
            # Again, 1 as the output as it is adversarial and if generator did a great
            #job of folling the discriminator then the output would be 1 (true)
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            # Print losses on this batch
            print('Epoch>{:.2f}, Batch {}/{}, d1={:.3f}, d2={:.3f}, g={:.3f}'.format(i + 1, j + 1, bat_per_epo, d_loss_real, d_loss_fake, g_loss[0]))

    # save the generator model
    g_model.save('cifar_generator_2epochs.h5')
