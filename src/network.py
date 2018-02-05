import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import mnist_loader

training_data, training_labels, validation_data, validation_labels, test_data, test_labels = mnist_loader.load_data_wrapper()

model = Sequential([
    Dense(30, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(training_data, training_labels,
          epochs=1,
          batch_size=10)