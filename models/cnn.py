import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


# Load saved subsets
# X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn


# Model and training parameters
n_labels = y_train_mlp.shape[1]
n_inputs_cnn = X_train_cnn.shape[1]*X_train_cnn.shape[2]
neur_cnn = 400 # neurons in fully-connected layer of cnn
frames = 41
cnnbands = 60
n_channels = 2
k = 'TruncatedNormal' # kernel initializer
batch_size = 100
epochs = 1000
lr_mlp = 1e-4 # learning rate


# Early stopping is used to provide guidance as to how many iterations can be run 
# before the learners begin to over-fit.
earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')


def build_cnn():
    
    '''
    A simple architecture with strong regularization is employed. 
    For training, cross entropy with AdamOptimizer is used, 
    given that this is a classification problem with a categorical set of targets.

    Note that Batch Normalization was placed after non-linearities 
    because that gave the best performance on validation sets.
    '''
    	
    model = Sequential()
    
    '''
    48 filters: hidden units which take advantage of the local structure present in the input data.
    (4,4) filter size: process a 4x4 block of pixels of the input space.
    Zero-padding: pad the input volume with zeros around the border.
    (2,2) stride: slide the filters over the input space 2x2 pixels at a time.
    Truncated initialization for weights with L2 regularization.
    (2,2) max pooling: downsampling the number of parameters by taking the max of a 2x2 subregion.
    ReLu activation: apply the non-linear function max(0,x) for each input x. 
    Batchnorm: scales the inputs to have zero mean and unit variance.
    '''
	
    model.add(Conv2D(filters=48, 
                     kernel_size=(4,4), 
                     padding='same',
                     strides=(2,2),
                     kernel_initializer=k,
                     kernel_regularizer=l2(0.01),
                     input_shape=(cnnbands, frames, n_channels)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    # next layer has 96 convolution filters
    model.add(Conv2D(filters=96, 
                     kernel_size=(4,4),
                     strides=(2,2),
                     padding='same',
                     kernel_initializer=k,
                     kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))    
    model.add(BatchNormalization())
    
    # flatten output into a single dimension 
    model.add(Flatten())

    # a FC layer learns non-linear combinations of the features outputted above
    model.add(Dense(neur_cnn, kernel_initializer=k, W_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # finally the output layer
    model.add(Dense(n_labels, kernel_initializer=k))
    model.add(Activation('softmax'))

    # optmization
    adam = Adam(lr=lr_cnn)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    
    return model

cnn = build_cnn()

# Training the CNN
trained_cnn = cnn.fit(X_train_cnn, 
                      y_train_cnn, 
                      validation_data=(X_test_cnn, y_test_cnn), 
                      callbacks=[earlystop],
                      batch_size=batch_size, 
                      epochs=epochs, 
                      verbose=1) # will display progress on every epoch
