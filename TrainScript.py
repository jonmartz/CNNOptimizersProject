import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
import os
import time




# ----------------- #
# global parameters #
# ----------------- #

rel_path = "C:\\Users\\guy schlesinger\\PycharmProjects\\CNNOptimizersProject\\results\\saves"
dataset_size = 10000
epochs = 15
batch_size = 128
learning_rate = 0.001
n_folds = 5
validation_split = 1/n_folds
optimizers_names = [
    'Adam',
    'NAdam',
    'RMSprop'
]
dataset_dic = {'mnist_digits' : tf.keras.datasets.mnist.load_data, 'mnist_fashion' : tf.keras.datasets.fashion_mnist.load_data}
optimizers = {
    'Adam': keras.optimizers.Adam(learning_rate=learning_rate),
    'NAdam': keras.optimizers.Nadam(learning_rate=learning_rate),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=learning_rate)

}
histories = {}
metrics = ['accuracy', 'loss']
run_time = {}
# For each data set
for dataset_name in dataset_dic:

    # Create a directory for the results
    path = "%s\\%s" %(rel_path,dataset_name)
    try:
        if not os._exists(path):
            os.mkdir(path)
    except OSError:
        path = path

    print('datatset name - %s' % dataset_name)

    # Downloading the Data
    (x_train, y_train), (x_test, y_test) = dataset_dic[dataset_name]()

    # Creating k fold experiment
    x = np.concatenate((x_train, x_test), axis=0)[:dataset_size]
    y = np.concatenate((y_train, y_test), axis=0)[:dataset_size]
    kf = KFold(n_folds,shuffle=True,random_state=42)
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x = x.reshape(x.shape[0], 28, 28, 1).astype('float32')
    input_shape = (28, 28, 1)

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x /= 255

    #For each optimizer
    for optimizer_name in optimizers_names:
        print('\noptimizer: %s' % optimizer_name)
        fold = 0
        # For each fold
        start = time.time()
        for train, test in kf.split(x):
            fold = fold +1
            print("fold %d" % fold)
            optimizer = optimizers[optimizer_name]

            # Create train/test dataset
            x_train = x[train]
            y_train = y[train]
            x_test = x[test]
            y_test = y[test]
            print("train size - %d , test size - %d" % (len(x_train),len(x_test)))

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


            history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size,
                                verbose=2, shuffle=False)
            histories["%s_%s_%d" %(dataset_name,optimizer_name,fold)] = history

            # save model for future training or testing
            model.save('%s\\%s\\%s_%d.h5' % (rel_path,dataset_name,optimizer_name,fold))

            # save training to csv
            df = pd.DataFrame(history.history)
            df.index.name = 'epoch'
            df.to_csv('%s\\%s\\%s_%d.h5' % (rel_path,dataset_name,optimizer_name,fold))


            # Testing model on a specific instance
            image_index = random.randrange(len(y_test))
            pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
            print('prediction = %d (should be %d)' % (pred.argmax(), y_test[image_index]))
            #plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
            #plt.show()
        end = time.time()
        run_time["%s_%s" % (dataset_name,optimizer_name)] = end - start

# plot for all optimizers and datasets
for dataset_name in dataset_dic:
    first = True
    for metric in metrics:
        for optimizer_name in optimizers_names:
            if first:
                rt = run_time["%s_%s" % (dataset_name, optimizer_name)]
                print("\nThe run time for %s optimizer in %s dataset is %s and average (per folds and epoch) - %s " % (optimizer_name, dataset_name, rt, rt / (n_folds*epochs)))
            metric_res = [0]*epochs
            rt = run_time["%s_%s" % (dataset_name,optimizer_name)]
            # Averaging the results over all folds
            for fold in range(1,n_folds+1):
                history = histories["%s_%s_%d" %(dataset_name,optimizer_name,fold)]
                for ep in range(epochs):
                    metric_res[ep] = metric_res[ep] + history.history['%s' % metric][ep]
            for ep in range(epochs):
                metric_res[ep] = metric_res[ep] / n_folds

            plt.plot(range(1, epochs + 1), metric_res, label=optimizer_name)
        first = False
        plt.title('%s comparison in %s dataset' % (metric,dataset_name))
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.xticks(range(1, epochs + 1))
        plt.legend()
        # plt.grid()
        plt.savefig('%s\\%s\\%s comparison.png' % (rel_path,dataset_name,metric))
        plt.show()
