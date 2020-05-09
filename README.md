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

### Test your own Image
```
# Additional components/libraries
import scipy
from PIL import Image
from scipy import ndimage

my_image = "your_image.jpg"
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
image = image/255.
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("The algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
```
