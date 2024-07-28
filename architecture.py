
from tensorflow.keras.layers import Input,Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from  tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


''' 
This is the Discriminator component.
This is trained individually therefore we compile it.
(JUST A BINARY CLASSIFIER!)
'''
def define_discriminator(in_shape=(32,32,3)):
	model = Sequential()
	model.add(Input(shape = in_shape)) 
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same')) #16x16x128
	model.add(LeakyReLU(negative_slope=0.2))
	
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same')) #8x8x128
	model.add(LeakyReLU(negative_slope=0.2))
	
	model.add(Flatten()) #shape of 8192
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid')) #shape of 1
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model


'''
This is the Generator component.
Takes a latent vector and outputs an image
'''

# We will reshape input latent vector into 8x8 image as a starting point. 
#So n_nodes for the Dense layer can be 128x8x8 so when we reshape the output 
#it would be 8x8x128 and that can be slowly upscaled to 32x32 image for output.
 #latent_dim is the dimension of the latent vector (e.g., 100)
 
def define_generator(latent_dim):   
    model = Sequential()
    n_nodes = 128 * 8 * 8  #8192 nodes
    model.add(Input(shape = (latent_dim,)))
    model.add(Dense(n_nodes)) #Dense layer so we can work with 1D latent vector
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Reshape((8, 8, 128)))  #8x8x128 dataset from the latent vector. 
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) #16x16x128
    model.add(LeakyReLU(negative_slope=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) #32x32x128
    model.add(LeakyReLU(negative_slope=0.2))
    # generate
    model.add(Conv2D(3, (8,8), activation='tanh', padding='same')) #32x32x3
    return model  #Model not compiled as it is not directly trained like the discriminator.
                    #Generator is trained via GAN combined model. 
                    
                    
    
''' 
Let's create the GAN:
When we train the Generator the discriminator is set to not trainable
'''
def define_gan(generator, discriminator):
	discriminator.trainable = False  #Discriminator is trained separately. So set to not trainable.
	# connect generator and discriminator
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model




