from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import random as rd
import matplotlib.pyplot as plt
from keras.layers import Input


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)
classes = 10
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)
input_size = 3072
batch_size = 100
epochs = 11
model = Sequential()
model.add(Input(shape=(input_size,)))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1)

fig = plt.figure()
outer_grid = gridspec.GridSpec(10, 10, wspace=0.0, hspace=0.0)
weights = model.get_weights()
w=weights[0].transpose()
for i , neuron in enumerate(rd.sample(range(0,1023), 100)):
    ax = plt.Subplot(fig, outer_grid[i])
    ax.imshow(np.mean(np.reshape(w[i], (32, 32, 3)),axis=2), cmap=cm.Greys_r)
    ax.set_xticks([])
    ax.set_yticks([])
fig.add_subplot(ax)
fig.show()

