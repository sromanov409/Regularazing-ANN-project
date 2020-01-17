import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.datasets import mnist
from keras import backend 
from keras import regularizers
import keras


# input image dimensions
img_rows, img_cols = 28, 28
#Number of classes
classes = 10
#Load the MNIST dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Tranform the data.
if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_whole = np.concatenate((x_train,x_test))

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)
y_whole = np.concatenate((y_train,y_test))

#Number of folds.
n_splits = 5
#Initialise the cross validaion.
kfold = KFold(n_splits, shuffle=True)
#Used to store the scores over the folds.
cvscores= []
#Used to store accuracies over epochs.
histories = []
fold_number = 1
for train, test in kfold.split(x_whole):
    #Setup the model.
    print('Fold :' +str(fold_number))
    fold_number += 1
    #Basic model parameters without any regularization. Appropriate reguralizers can be commented in.
    cnn = Sequential()
    cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape,))
# =============================================================================
#     cnn.add(Conv2D(32, kernel_size=(3, 3),activation='relu',
#                      input_shape=input_shape,activity_regularizer=regularizers.l2(0.001)))
# =============================================================================
    cnn.add(BatchNormalization(axis = 3))
    
    #cnn.add(Conv2D(64, (3, 3), activation='relu',activity_regularizer=regularizers.l2(0.001)))
    cnn.add(Conv2D(64, (3, 3), activation='relu'))
    cnn.add(BatchNormalization(axis  = 3))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    
    #cnn.add(Dense(128, activation='relu',activity_regularizer=regularizers.l2(0.001)))
    cnn.add(Dense(128, activation='relu'))
    cnn.add(BatchNormalization())
    #cnn.add(Dropout(0.5))
    
    cnn.add(Dense(classes, activation='softmax'))
    cnn.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    #Fit the model using specifyed hyper parameters.
    history = cnn.fit(x_whole[train], y_whole[train],
              batch_size=100,
              epochs=6,
              verbose=1,
              validation_data=(x_test, y_test))
    #Evaluate the models's performance.
    score = cnn.evaluate(x_whole[test], y_whole[test], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    cvscores.append(score)
    histories.append(history)
    
#Plot the results.
plt.title('Fold accuracy and val_accuracy over epochs.')
plt.plot(histories[0].history['accuracy'], label='Train Accuracy Fold 1', color='black')
plt.plot(histories[0].history['val_accuracy'], label='Val Accuracy Fold 1', color='black', linestyle = "dashdot")
plt.plot(histories[1].history['accuracy'], label='Train Accuracy Fold 2', color='red', )
plt.plot(histories[1].history['val_accuracy'], label='Val Accuracy Fold 2', color='red', linestyle = "dashdot")
plt.plot(histories[2].history['accuracy'], label='Train Accuracy Fold 3', color='green', )
plt.plot(histories[2].history['val_accuracy'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
plt.plot(histories[3].history['accuracy'], label='Train Accuracy Fold 4', color='blue', )
plt.plot(histories[3].history['val_accuracy'], label='Val Accuracy Fold 4', color='blue', linestyle = "dashdot")
plt.plot(histories[4].history['accuracy'], label='Train Accuracy Fold 5', color='orange', )
plt.plot(histories[4].history['val_accuracy'], label='Val Accuracy Fold 5', color='orange', linestyle = "dashdot")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['acccuracy','val_accuracy'], loc = 'lower right')
plt.show()
#plt.savefig('MNIST_BN_DP.png')
print(cvscores)


#Save the scores over folds in a binary file.
import pickle
filehandler = open("MNIST_all","wb")
pickle.dump(cvscores,filehandler)
filehandler.close()


# =============================================================================
# [[0.035952037941980444, 0.9904285669326782], [0.02755246050185607, 0.990928590297699], [0.029663836435320265, 0.991357147693634], [0.03496182792725657, 0.9909999966621399], [0.028242351643283266, 0.9915714263916016]]
# =============================================================================



