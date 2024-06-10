# Deep-Learning-from-Scratch
An attempt to recreate the most basic features of tensorflow and keras from scratch, using the numeric python library numpy. 

This framework has been created for educational purposes only.

## Installation
```
pip install dlfs
```

## Usage
The usage is very similar to the keras library. 

```python
from dlfs.models import Sequential
from dlfs.layers import Dense, Dropout, Conv2D, Flatten
from dlfs.optimizers import SGDMomentum

model = Sequential()
model.add(Conv2D(1, (3, 3), activation='relu', input_shape=(28, 28, 1), convolution_type='simple'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=SGDMomentum(learning_rate=0.1),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=100, verbose=2, 
                    validation_data=(X_test, y_test))
```

For a complete example see:
[dlfs_examples.pynb](https://colab.research.google.com/drive/1EbiSrOFxtqygvK_wsc41_su8_oUTPL3J?usp=sharing).
### Install dependencies
You can do this easily if needed thanks to the requirements.txt file by just running the following command:
```
pip install -r requirements.txt
```

### Python version
Everything is written in Python 3.7 (to be compatible with Google Colab).

## License
This project is licensed under the Apache 2.0 license (see 
[LICENSE.md](https://github.com/Pabloo22/Deep-Learning-from-Scratch/blob/main/LICENSE)).

## Acknowledgments
This project was inspired by the Tensorflow and [Keras](https://keras.io/) framework.
