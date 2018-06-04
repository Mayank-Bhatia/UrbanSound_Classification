import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# Load saved subsets
# X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn


# Model and training parameters
n_labels = y_train_mlp.shape[1]
rnnbands = 20
n_steps = 41
k = 'TruncatedNormal' # kernel initializer
batch_size = 100
epochs = 1000
lr_mlp = 1e-4 # learning rate


# Early stopping is used to provide guidance as to how many iterations can be run 
# before the learners begin to over-fit.
earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')


def build_rnn():

    '''
    A simple architecture with strong regularization is employed. 
    For training, cross entropy with AdamOptimizer is used, 
    given that this is a classification problem with a categorical set of targets.

    Note that Batch Normalization was placed after non-linearities 
    because that gave the best performance on validation sets.
    '''
    
    model = Sequential()
    model.add(LSTM(units=300, # transforms input sequence into a single vector of size 300
                   dropout=0.1, # for input->output
                   recurrent_dropout=0.3, # for time-dependant units
                   return_sequences=True, # many-to-many
                   input_shape=(rnnbands, n_steps)))
    model.add(BatchNormalization())
    model.add(LSTM(units=75, 
                   dropout=0.1, 
                   recurrent_dropout=0.3, 
                   return_sequences=False)) # many-to-one
    model.add(BatchNormalization())
    model.add(Dense(n_labels, 
                    kernel_initializer=k, # truncated normal distribution
                    activation='softmax'))
    
    # optimzation
    adam = Adam(lr=lr_rnn, epsilon=1e-06)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    
    return model

rnn = build_rnn()


# Training the RNN
trained_rnn = rnn.fit(X_train_rnn, 
                      y_train_rnn, 
                      validation_data=(X_test_rnn, y_test_rnn), 
                      callbacks=[earlystop],
                      batch_size=batch_size, 
                      epochs=epochs, 
                      verbose=1) # will display progress on every epoch
