# -*- coding: utf-8 -*-
from numpy import genfromtxt
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt
import random

dataset = genfromtxt('data/SpamDataset/spambase.data', delimiter=',')
random.shuffle(dataset)
training_set = dataset[:3000]
test_set = dataset[3000:]

model = models.Sequential()
model.add(layers.Dense(3, activation='sigmoid', input_shape=(48,)))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer=optimizers.RMSprop(lr=0.05),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

x_train = training_set[:, :48]
y_train = training_set[:, -1]

x_val = x_train[:500]
partial_x_train = x_train[500:]

y_val = y_train[:500]
partial_y_train = y_train[500:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=15,
                    batch_size=500,
                    validation_data=(x_val, y_val))


acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

print(model.evaluate(test_set[:,:48],test_set[:,-1]))

