# TensorFlow Signal Processing

##### Joel Shapiro

# problem overview

The objective of this project is to evaluate the effectiveness of doing audio effect emulation using deep learning. For audio, there are two main classifications for tools: generators and effects. A generator is something which takes non-audio input, either physical or midi, and creates audio out of it. This would include instruments, synthesizers, drums - basically anything that really stands out as being traditionally musical. The other category, effects, are elements which take audio as input and transform it into some other audio to output. This can range from a simple filter to more complex effects such as distortion or reverb; even the echo of a room or quality drop over a phone call is an effect. The idea behind this project is to see if we can’t train a network to emulate one of these effects using deep learning.  
Audio is an interesting medium to work in for machine learning, as like image data the output can be judged both quantitatively and qualitatively. On top of this, audio itself is a complex structure: the additive property of waves can cause some unforeseen outcomes. On top of that, digital audio data is inherently convoluted: it is stored as a time series of points which are sampled from the audio signal itself. These points are fast fourier transformed back into the signal whenever the audio is ready to be output. Because of this, a lot of the information which is affected by effects is hidden behind this signal processing problem.  
In the past, doing signal processing in machine learning involved doing some manual decomposition of the input in order to abstract away the signal processing [1]. Often audio would be rendered into images of the spectrogram, which show the frequency distribution of the audio. While this works well for classification problems, it lacks accuracy for end to end regression problems like ours. For that, we need to do actual signal processing in order to detect the features that matter.  

The current progress on this project is available at [github.com/jshap70/TensorFlow-Signal-Processing](http://github.com/jshap70/TensorFlow-Signal-Processing)


## sample types

Previously I mentioned how audio is a conceptionally complex structure. This is because audio data is time series data of the amplitute of the audio, however all of the information that is "stored" in it is 

<img src="https://github.com/jshap70/TensorFlow-Signal-Processing/raw/master/resources/microcontrollers_fft_example.png" width="550" alt="fourier transforms and signals"> [2]


Sound can also have harmonics 

<!--<img src="https://rawcdn.githack.com/jshap70/TensorFlow-Signal-Processing/master/resources/chello_frequency.svg" width="550" alt="frequency of chello">-->
<img src="./resources/chello_frequency.svg" width="550" alt="frequency of chello"> [3]




<img src="https://github.com/jshap70/TensorFlow-Signal-Processing/raw/master/resources/sample-rate.png" height="250" alt="point sampling in digital audio"> [4]

The audio used in this project has a uniform sample rate, meaning that we're ensuring that we don't have to worry about TensorFlow actually understanding that the data inputs are time dependent.   
The audio is mostly composed of some simple, generated audio samples which covers a varying types of sound. On the more simple side, we have simple sine, triangle, and saw waves that move through a frequency range. More difficult samples include piano recordings and voice data. The scope of this project was only on simple effects because of the time and resources available, however it would be interesting to see the impact filter complexity has on training difficulty.



# the network

Starting off I used a standard, fully connected regression neural network with varying depths of hidden layers. The goal of this network was to try to overfit the training data to show that it can at least be brute forced.

Because this problem is attempting to directly emulate a filtering effect, it seems somewhat intuitive that this problem would be decently well represented by a convolutional network. If we could get the neural net to understand the audio input, we could train a convolutional layer to understand the change in the data from the input to the output. This might also allow us to combine convolutional layers which are trained from different filters, but more on that later.


## batching and data sampling

Looking at the data itself, the wav files are stereo 16bit PCM (integer)) files. To begin with, I converted the data to a 32bit float wav file and normalized the audio to fit within that standard. I converted the data to numpy arrays, keeping the audio in stereo. I could have split apart each file into mono tracks, effectively doubling the amount of data I would be training on, but I chose not to because some effects might have stereo-dependent elements to them.

Before we begin batching, validation and testing data is extracted prior to the batching to ensure that it is not trained on at all. Because it is time series data the batching is a bit trickier. Although the data needs to be kept in contiguous chunks, we can still extract smaller sections of it to train on so we don’t have to train on the whole dataset at one time. I implemented a batching system that does a scrolling window selection of the audio for discrete time periods. If the offset that each window is from each other is smaller than the length of each sample, then there will be some overlap of the batches. Initially, I have set up up the batches in a way that means every datapoint will be used 10 times. After generating the batches I shuffle them out of order so that we can get a more even distribution.




## future plans

// RNN’s and how they might help with time series data



### Notes / Sources

[1] At least this is true for most practical applications. An example can be seen here: [github.com/markostam/audio-deepdream-tf](https://github.com/markostam/audio-deepdream-tf)

[2] Image showing the relationship between time series and frequency data. Source: [learn.adafruit.com/fft-fun-with-fourier-transforms/background](https://learn.adafruit.com/fft-fun-with-fourier-transforms/background)

[3] images showing the relationship between amplitute and frequency. Source: https://processing.org/tutorials/sound/

[4] image showing how digital audio data is stored. Source: [progulator.com/digital-audio/sampling-and-bit-depth/](http://progulator.com/digital-audio/sampling-and-bit-depth/) however, note that there are some very large errors in this article. Most importantly, it incorrectly does not cover how fourier transforms are used to go from the digital point sampling back to the analog signal. 


[misc]  
some audio midi from - http://www.piano-midi.de/brahms.htm  
wavio - https://github.com/mgeier/python-audio/ - by Warren Weckesser