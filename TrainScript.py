import keras
import keras_radam
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# ----------------- #
# global parameters #
# ----------------- #
dataset_size = 5000
validation_split = 0.2
epochs = 5
batch_size = 128
learning_rate = 0.001
optimizers_names = [
    'Adam',
    'NAdam',
    'RMSprop',
    # 'RAdam'
]

# Downloading the Mnist Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x = np.concatenate((x_train, x_test), axis=0)[:dataset_size]
y = np.concatenate((y_train, y_test), axis=0)[:dataset_size]

# todo: implement k-fold cross validation

# Reshaping the array to 4-dims so that it can work with the Keras API
x = x.reshape(x.shape[0], 28, 28, 1).astype('float32')
input_shape = (28, 28, 1)

# Normalizing the RGB codes by dividing it to the max RGB value.
x /= 255

optimizers = {
    'Adam': keras.optimizers.Adam(learning_rate=learning_rate),
    'NAdam': keras.optimizers.Nadam(learning_rate=learning_rate),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=learning_rate),
    'RAdam': keras_radam.RAdam(min_lr=learning_rate, total_steps=10000, warmup_proportion=0.1)
}

histories = {}
metrics = ['accuracy', 'loss']

for optimizer_name in optimizers_names:
    optimizer = optimizers[optimizer_name]
    print('\noptimizer: %s' % optimizer_name)

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    # Compiling and Fitting the Model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x=x, y=y, validation_split=validation_split, epochs=epochs, batch_size=batch_size,
                        verbose=2, shuffle=False)
    histories[optimizer_name] = history

    # save model for future training or testing
    model.save('saves\\%s.h5' % optimizer_name)

    # save training to csv
    df = pd.DataFrame(history.history)
    df.index.name = 'epoch'
    df.to_csv('saves\\%s train.csv' % optimizer_name)

    # # plot for this optimizer
    # for metric in metrics:
    #     plt.plot(history.history[metric])
    #     plt.plot(history.history['val_%s' % metric])
    #     plt.title('%s %s' % (optimizer_name, metric))
    #     plt.ylabel(metric)
    #     plt.xlabel('epoch')
    #     plt.legend(['train', 'test'], loc='upper left')
    #     plt.grid()
    #     plt.show()

    # Testing model on a specific instance
    image_index = 4444
    pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
    print('prediction = %d (should be %d)' % (pred.argmax(), y_test[image_index]))  # should be 9
    # plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
    # plt.show()

# plot for all optimizers
for metric in metrics:
    for optimizer_name in optimizers_names:
        history = histories[optimizer_name]
        plt.plot(range(1, epochs + 1), history.history['val_%s' % metric], label=optimizer_name)
    plt.title('%s comparison' % metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    # plt.grid()
    plt.savefig('saves\\%s comparison.png' % metric)
    plt.show()
