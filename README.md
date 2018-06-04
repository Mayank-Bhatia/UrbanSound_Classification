The [UrbanSound8k](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html) dataset 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, 
children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, and street_music.

We'll take a look at a standard multiperceptron model, as well as convolutional and recurrent nets. Feature extraction is done using
the open source library called [Librosa](http://librosa.github.io/). Librosa allows you to load sound files, extract features, generate waveplot, and much more. 

## Requirements:
librosa==0.6.0 <br>
pandas==0.20.3 <br>
Keras==2.1.5 <br>
numpy==1.14.2+mkl <br>
scikit_learn==0.19.1 <br>

## Acknowledge:
J. Salamon, C. Jacoby and J. P. Bello, "[A Dataset and Taxonomy for Urban Sound Research]", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.
[[ACM](https://dl.acm.org/citation.cfm?id=2655045)][[PDF](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf)][[BibTeX](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.bib)]




