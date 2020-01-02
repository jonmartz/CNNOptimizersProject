import keras
import keras_radam
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# ----------------- #
# global parameters #
# ----------------- #
epochs = 1
train_fraction = 0.8
learning_rate = 0.001
optimizers_names = [
    'Adam',
    # 'NAdam',
    'RMSprop',
    # 'RAdam'
]

# Downloading the Mnist Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_all = np.concatenate((x_train, x_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
# print('x_train shape:', x_train.shape)
# print('Number of images in x_train', x_train.shape[0])
# print('Number of images in x_test', x_test.shape[0])

optimizers = {
    'Adam': keras.optimizers.Adam(learning_rate=learning_rate),
    'NAdam': keras.optimizers.Nadam(learning_rate=learning_rate),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=learning_rate),
    'RAdam': keras_radam.RAdam(min_lr=learning_rate, total_steps=10000, warmup_proportion=0.1)
}

for optimizer_name in optimizers_names:
    optimizer = optimizers[optimizer_name]
    print('optimizer: %s' % optimizer_name)

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    # Compiling and Fitting the Model
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x=x_train, y=y_train, epochs=epochs)

    # Evaluating the Model
    model.evaluate(x_test, y_test)

    image_index = 4444
    plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
    pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
    print(pred.argmax())  # should be 9
    plt.show()
