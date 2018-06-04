import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


# Load saved subsets
# X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp


# Model and training parameters
n_labels = y_train_mlp.shape[1]
n_inputs_mlp = X_train_mlp.shape[1]
neur_mlp1 = 500 # neurons in fully-connected layer of mlp
neur_mlp2 = 1000
k = 'TruncatedNormal' # kernel initializer
batch_size = 100
epochs = 1000
lr_mlp = 1e-4 # learning rate


# Early stopping is used to provide guidance as to how many iterations can be run 
# before the learners begin to over-fit.
earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')


def build_mlp():

    '''
    A simple architecture with strong regularization is employed. 
    For training, cross entropy with AdamOptimizer is used, 
    given that this is a classification problem with a categorical set of targets.
	
    Note that Batch Normalization was placed after non-linearities 
    because that gave the best performance on validation sets.
    '''
    
    model = Sequential()
    model.add(Dense(n_inputs_mlp, input_dim=n_inputs_mlp))
        
    # first fully connected layer
    model.add(Dense(neur_mlp1, kernel_initializer=k, W_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # second fully connected layer
    model.add(Dense(neur_mlp2, kernel_initializer=k, W_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # output layer with one node per class
    model.add(Dense(n_labels, kernel_initializer=k))
    model.add(Activation('softmax'))

    # optimization
    adam = Adam(lr=lr_mlp, epsilon=1e-6)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    
    return model

mlp = build_mlp()


# Training the MLP
trained_mlp = mlp.fit(X_train_mlp, 
                      y_train_mlp, 
                      validation_data=(X_test_mlp, y_test_mlp), 
                      callbacks=[earlystop],
                      batch_size=batch_size, 
                      epochs=epochs, 
                      verbose=1) # will display progress on every epoch
