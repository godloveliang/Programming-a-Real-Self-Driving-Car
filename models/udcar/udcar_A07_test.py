#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
#import pickle
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.models import load_model
import pickle
import sys
import numpy as np
#from sklearn.model_selection import train_test_split
import os
import random


# CONFIGURABLE PARAMETERS
test_model = 'tl_keras_carAv07.h5'
image = "image.jpg"
BATCH_SIZE = 8
IMAGE_HEIGHT = 600
IMAGE_WIDTH = 800
IMAGE_CHANNEL=3
red_path = "data/udcarimg/Red"
yellow_path = "data/udcarimg/Yellow"
green_path = "data/udcarimg/Green"
unknown_path = "data/udcarimg/Unknown"
file_type = "*.jpg"
test_model = 'tl_keras_carAv07.h5'
#brt_factor = 0.2



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
    #print('Image shape', allimg[0].shape)
    
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

def load_images(allimg, allcol):
    #print("Loading images")
    X = np.empty([0, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], dtype=np.float32)
    y = np.empty([0], dtype=np.int32)

    for i in range(len(allimg)):
        # get an image and its corresponding color for an traffic light
        color = allcol[i]
        image = cv2.imread(allimg[i])
        image = image[:,:,::-1]
        image = img_bright(image,brt_factor)
        image = img_flip(image)
        
        # Appending them to existing batch
        X = np.append(X, [image], axis=0)
        y = np.append(y, [color])
    #y = to_categorical(y, num_classes=4)
    return X,y
        

# Predict classes for a set of images and return accuracy
def predict_classes(Xvalid, yvalid):
    #print("Predicting classes")
    #model = load_model(test_model)
    ypred = model.predict_classes(Xvalid)
    acc = np.mean(ypred==yvalid)
    print("Validation accuracy", acc)
    #print("Average of labels", np.mean(yvalid))
    return


# Predict traffic light for a specific image
def predict_signal(image):
    imgdisp = cv2.imread(image)
    imgdisp = imgdisp[:,:,::-1] # BGR to RGB
    
    imgpred = imgdisp.reshape((1,imgdisp.shape[0], imgdisp.shape[1], imgdisp.shape[2]))
    classpred = model.predict_classes(imgpred)
    print('Predicted traffic light: ', labelsdict[classpred[0]])
    return

if __name__ == "__main__":
    check_versions()
    model = load_model(test_model)
    allimg, allcol = list_images()
    allimgs, allcols = shuffle_lists(allimg, allcol)
    #bright_list = [ 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    bright_list = [1]
    for brt_factor in bright_list:
        print("************************************************")
        print("Brightness factor is:", brt_factor)
        Xvalid, yvalid = load_images(allimgs[:100], allcols[:100])
        predict_classes(Xvalid, yvalid)
    