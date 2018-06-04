import os
import glob
import numpy as np
import pandas as pd
import librosa as ls
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer


# path to dataset
sound_dir = 'UrbanSound8K\\audio\\files'
# list of all .wav files in sound_dir
sounds = glob.glob(sound_dir + '\\*.wav')


def windows(data, window_size):
    
    ' Generate equal-sized segments from the sounds (for CNN and RNN). '
    ' Sounds with a longer duration generate more features. '
    
    start = 0
    while start < len(data):
        yield start, start + window_size
        # For a window-size of 1024, hop-length of 512 is needed.
        start += (window_size / 2)

def make_features(data, cnnbands=60, rnnbands=20, frames=41, window_size=512*40):
    
    ' Extract features from sound files. '
    ' Three different types of features are extracted: '
    ' MLP: chroma, mfcc, tonnetz, melspecgram '
    ' CNN: segments of melspecgram '
    ' RNN: segments of mfcc '
    
    try:      
        # Load and decode the audio as:
        # 1) time series ts - a 1D array
        # 2) variable sr - sampling rate of ts (samples per second of audio).
        ts, sr = ls.load(data, res_type='kaiser_fast') 
        
        # Compute the short-time Fourier transform (stft):
        # 1) divide a longer time signal into shorter segments of equal length
        # 2) map to a sinusoidal basis.
        # Output is a matrix D(f,t) such that |D(f,t)| is the magnitude of frequency bin f at time t.
        stft = np.abs(ls.stft(ts))

        # Note that for "mel" and "mfcc" features, spectra are mapped to the mel basis.
        # This approximates the mapping of frequqncies to patches of nerves in the cochlea.
        # Doing so lets you use spectral information in about the same way as human hearing.
        
        # Project the entire spectrum onto 12 bins representing the 12 semitones of the musical octave.
        # Then take the mean along each bin to generate 12 features.
        chroma = np.mean(ls.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        
        # Compute the mel-frequency cepstral coefficients (MFCC):
        # 1) map the powerspectrum of the stft onto the mel scale (pitch instead of frequency).
        # 2) take log of the powers at each frequency.
        # 3) take a cosine transform of these log powers.
        # 4) MFCCs are the amplitudes of the resulting spectrum, shape=[n_mfcc,time].
        # Then take the mean along the first axis, generating 40 features.
        mfcc = np.mean(ls.feature.mfcc(y=ts, sr=sr, n_mfcc=40).T, axis=0)
        
        # Compute tonal centroid features by transforming the 12 bin chromas to 6D vectors.
        # Then take the mean along each dimension to generate 6 features. 
        tonnetz = np.mean(ls.feature.tonnetz(y=ls.effects.harmonic(ts), sr=sr).T, axis=0)
        
        # Compute the spectogram of the stft and map to the mel scale, shape=[n_mels,time].
        # Then take the mean along the first axis, generating 128 features.
        mel = np.mean(ls.feature.melspectrogram(ts, sr=sr).T,axis=0)
        
        # So far the total number of features is 12+40+6+128 = 186 features. 
        # These will be the features used in the Multilayer Perceptron model.
        mlpfeatures = np.concatenate([chroma,mfcc,tonnetz,mel])

        # For CNN and RNN, features are extracted from equal-sized segments of the sound clips.
        # CNN features are just the mel spectrograms (based off the paper linked above).
        # RNN features are the MFCCs (in an effort to try different features than CNN's).
        
        # CNN features
        specgrams = []
        for start,end in windows(ts,window_size):
            start = int(start)
            end = int(end)
            if len(ts[start:end]) == window_size :
                signal = ts[start:end]
                melspec = ls.feature.melspectrogram(signal, n_mels=cnnbands)
                melspec = ls.amplitude_to_db(melspec) # logarithmic units
                melspec = melspec.T.flatten()[:, np.newaxis].T # expand dim to help with reshaping
                specgrams.append(melspec)
        specgrams = np.asarray(specgrams).reshape(len(specgrams),cnnbands,frames,1)
        cnnfeatures = np.concatenate((specgrams, np.zeros(np.shape(specgrams))), axis=3)
        for i in range(len(cnnfeatures)):
            # Estimate first-order derivatives of each feature dimension.
            # This will be the second channel for CNN features.
            cnnfeatures[i, :, :, 1] = ls.feature.delta(cnnfeatures[i, :, :, 0])

        # RNN features
        mfccs = []
        for start,end in windows(ts,window_size):
            start = int(start)
            end = int(end)
            if len(ts[start:end]) == window_size :
                signal = ts[start:end]
                rnn_mfcc = ls.feature.mfcc(y=signal, sr=sr, n_mfcc=rnnbands)
                rnn_mfcc = rnn_mfcc.T.flatten()[:, np.newaxis].T
                mfccs.append(rnn_mfcc)
        rnnfeatures = np.asarray(mfccs).reshape(len(mfccs),rnnbands,frames)

        return mlpfeatures,cnnfeatures,rnnfeatures
    
    except Exception as e:
        print("Error encountered with file:", data)


def make_labels(data):
    
    ' Extract target class from the filepath. '
    
    label = data.split('-')[1]
    return int(label)


# Parsing each sound file to extract features and labels
labels = []
mlpfeatures = []
cnnfeatures = []
rnnfeatures = []
for i in range(len(sounds)):
    audio_file = os.path.join(os.getcwd(), sounds[i])
    f = make_features(audio_file)
    if f is not None and len(f[0])>0 and len(f[1])>0 and len(f[2])>0:
        labels.append(make_labels(audio_file))
        mlpfeatures.append(np.array(f[0]))
        cnnfeatures.append(np.array(f[1]))
        rnnfeatures.append(np.array(f[2]))
labels = np.array(labels)
mlpfeatures = np.array(mlpfeatures)
cnnfeatures = np.array(cnnfeatures)
rnnfeatures = np.array(rnnfeatures)


# For CNN and RNN features, depending on the duration, there will be a longer (or shorter) 
# spectogram for each audio clip. we can get many samples for a file, all with the same label. 
# Let's split them to get separate data points.
labelsFULL = []
cnnfeaturesFULL = []
rnnfeaturesFULL = []
for idx,item in enumerate(cnnfeatures):
    full = np.split(item, item.shape[0])
    for k in full:
        cnnfeaturesFULL.append(np.squeeze(k))
        labelsFULL.append(labels[idx])
for idx,item in enumerate(rnnfeatures):
    full = np.split(item, item.shape[0])
    for k in full:
        rnnfeaturesFULL.append(np.squeeze(k))
		
		
# rescaling to avoid geometric bias towards any MLP features
scaler = MinMaxScaler(feature_range=(0,1))
mlpfeatures = scaler.fit_transform(mlpfeatures)
# one-hot encoding for the audio classes
encoder = LabelBinarizer()
mlplabels = encoder.fit_transform(labels)
cnnlabels = encoder.fit_transform(labelsFULL)
rnnlabels = cnnlabels