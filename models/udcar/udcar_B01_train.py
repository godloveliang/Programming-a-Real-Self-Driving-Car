#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.optimizers import Adam
from keras.layers import Lambda
from keras.layers import BatchNormalization
from keras.layers import Cropping2D


#import pickle
import sys
import numpy as np
import os
import random


# CONFIGURABLE PARAMETERS
BATCH_SIZE = 8
IMAGE_HEIGHT = 600
IMAGE_WIDTH = 800
IMAGE_CHANNEL=3
red_path = "data/udcarimg/Red"
yellow_path = "data/udcarimg/Yellow"
green_path = "data/udcarimg/Green"
unknown_path = "data/udcarimg/Unknown"
file_type = "*.jpg"
final_model = "tl_keras_carBv04.h5"

# Check versions
def check_versions():
    print("Checking versions")
    print(sys.version) #Python version
    print(tf.__version__) # TF version
    print(keras.__version__) #Keras version
    return

def list_images():
    print("Listing images")
    redsimg = glob.glob(os.path.join(red_path, file_type))
    yellowsimg = glob.glob(os.path.join(yellow_path, file_type))
    greensimg = glob.glob(os.path.join(green_path, file_type))
    unknownsimg = glob.glob(os.path.join(unknown_path, file_type))
    allimg = redsimg + yellowsimg + greensimg + unknownsimg
    allcol = [0] * len(redsimg) + [1] * len(yellowsimg) + [2] * len(greensimg) + [3] * len(unknownsimg)
    print('Category wise counts - red, yellow, green, unknown:', len(redsimg), len(yellowsimg), len(greensimg), 
          len(unknownsimg))
    print('Total # of Images', sum([len(redsimg), len(yellowsimg), len(greensimg), len(unknownsimg)]))
    return allimg, allcol

def shuffle_lists(allimg, allcol):
    np.random.seed(0)
    indices = np.arange(len(allimg))
    np.random.shuffle(indices)
    allimgs = [allimg[index] for index in indices]
    allcols = [allcol[index] for index in indices]
    return allimgs, allcols

# Image brightness reduction for an RGB image
def img_bright(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = factor*hsv[:, :, 2]
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img
#Flip images left to right
def img_flip(img):
    new_img = cv2.flip(img,1)
    return new_img


def generator(data,labels):
    while True:
        # Randomize the indices to make an array
        indices_arr = np.random.permutation(len(data))
        for batch in range(0, len(indices_arr), BATCH_SIZE):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + BATCH_SIZE)]

            # initializing the arrays, x_train and y_train
            X = np.empty(
                [0, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], dtype=np.float32)
            y = np.empty([0], dtype=np.int32)

            for i in current_batch:
                color = labels[i]
                image = cv2.imread(data[i])
                image = image[:,:,::-1]
                if (i%2==0):
                    brt_factor = random.uniform(0.5,1.0)
                    #brt_factor = 1.0
                    image = img_bright(image,brt_factor)
                if (i%3==0):
                    image = img_flip(image)
                    pass
        
                # Appending them to existing batch
                X = np.append(X, [image], axis=0)
                y = np.append(y, [color])
            y = to_categorical(y, num_classes=4)

            yield (X, y)

def create_traintest_gen(allimg, allcol):
    cutoff = int(0.8 * len(allimg)) 
    Xtrain , Xvalid = allimg[:cutoff], allimg[cutoff:]
    ytrain , yvalid = allcol[:cutoff], allcol[cutoff:]
    train_gen = generator(Xtrain, ytrain)
    validation_gen = generator(Xvalid, yvalid)
    return train_gen, validation_gen, cutoff

def train_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((0,300), (0,0)), input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Conv2D(32, (3, 3)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    #compile model using accuracy to measure model performance
    #model.compile(optimizer=Adam(1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_gen, steps_per_epoch = cutoff//BATCH_SIZE,epochs = 40, validation_data = validation_gen, validation_steps
                        = int(0.25*cutoff)//BATCH_SIZE, verbose = 1)
    model.save(final_model)
    return


# Predict classes for a set of images and return accuracy
def predict_classes(Xvalid, yvalid):
    #print("Predicting classes")
    #model = load_model(test_model)
    ypred = model.predict_classes(Xvalid)
    acc = np.mean(ypred==yvalid)
    print("Validation accuracy", acc)
    #print("Average of labels", np.mean(yvalid))
    return


if __name__ == "__main__":
    check_versions()
    allimg, allcol = list_images()
    allimgs, allcols = shuffle_lists(allimg, allcol)
    train_gen, validation_gen, cutoff = create_traintest_gen(allimgs, allcols)
    train_model()
    