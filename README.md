# TensorFlow-Reco

### Install TensorFlow
To install current release:
```
$ pip install tensorflow
```
To install a smaller CPU package:
```
$ pip install tensorflow-cpu
```

### TensorFlow Projects
- Implemented a deep neural network to recognize 0 to 5 in sign language using sign (training & test) datasets, with pretty impressive accuracy!
- Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).
- Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).
- Flatten training and test set images, normalize image vectors, convert the set labels to one-hot matrices (tensorflow.py).
- Built with [TensorFlow](https://github.com/tensorflow).
