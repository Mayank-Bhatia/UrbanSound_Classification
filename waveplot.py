import os
import numpy as np
import librosa as ls
import librosa.display as lsd
import matplotlib.pyplot as plt

# path to dataset
sound_dir = 'UrbanSound8K\\audio\\files'

# choosing one sound from each category
unique_sounds = ["57320-0-0-7.wav","24074-1-0-3.wav","15564-2-0-1.wav","31323-3-0-1.wav",
                 "46669-4-0-35.wav","89948-5-0-0.wav","40722-8-0-4.wav", 
                 "14772-7-5-0.wav","106905-8-0-0.wav","108041-9-0-4.wav"]
sound_names = ["air conditioner","car horn","children playing", "dog bark","drilling",
               "engine idling", "gun shot", "jackhammer","siren","street music"]

# raw time-series values for these sounds
raw = []
for u in unique_sounds:
    ts, sr = ls.load(os.path.join(sound_dir, u))
    raw.append(ts)

# waveplot of unique sound-classes within dataset
def waves(raw_sounds):    
    i = 1
    fig = plt.figure(figsize=(12,15))
    for file,name in zip(raw_sounds,sound_names):
        plt.subplot(10,1,i)
        lsd.waveplot(np.array(file), x_axis=None)
        plt.title(name.title())
        i += 1
    plt.suptitle("Waveplot", x=0.52, y=1.02, fontsize=15)
    fig.tight_layout()
    plt.savefig('waveplot.jpg')

waves(raw)
