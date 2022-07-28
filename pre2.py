import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle

def load_train_data():
    return zip(np.load('train_data.npy'), np.load('train_labels.npy'))

def load_test_data():
    return zip(np.load('test_data.npy'), np.load('test_labels.npy'))

if __name__ == "__main__":
    base_dir = 'chest_xray/'

    train_dir = base_dir + 'train/'
    test_dir = base_dir + 'test/'
    val_dir = base_dir + 'val/'

    train_neg = train_dir + 'NORMAL'
    train_pos = train_dir + 'PNEUMONIA'
    test_neg = test_dir + 'NORMAL'
    test_pos = test_dir + 'PNEUMONIA'
    val_neg = val_dir + 'NORMAL'
    val_pos = val_dir + 'PNEUMONIA'

    train_pos = [train_pos + '/' + i for i in os.listdir(train_pos)]
    train_neg = [train_neg + '/' + i for i in os.listdir(train_neg)]

    test_pos = [test_pos + '/' + i for i in os.listdir(test_pos)]
    test_neg = [test_neg + '/' + i for i in os.listdir(test_neg)]

    val_pos = [val_pos + '/' + i for i in os.listdir(val_pos)]
    val_neg = [val_neg + '/' + i for i in os.listdir(val_neg)]

    print('---------------------------------------------------')

    # size of smallest image to rescale each image into
    image_size = 127

    print('Building train data and train labels ...')

    train_full = train_pos + train_neg + val_pos + val_neg # Combining given train and validation datasets, (do your own train/validation splits later)

    train_data = []
    train_labels = []

    count = 0
    for train_img in train_full:
        img = cv2.imread(train_img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size, image_size))
        np_img = np.asarray(img)
        train_data.append(np_img)
        
        if "bacteria" in train_img or "virus" in train_img:
            train_labels.append(1)
        else:
            train_labels.append(0)

        if count % 750 == 0:
            print(f"{count} images processed")
        count += 1

    print('total number of images processed:', count)

    # ---------------------------------------------------
    print('---------------------------------------------------')
    print('Building test data and test labels ...')

    test_full = test_neg + test_pos

    test_data = []
    test_labels = []

    count = 0
    for test_img in test_full:
        img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size, image_size))
        np_img = np.asarray(img)
        test_data.append(np_img)
        if "bacteria" in test_img or "virus" in test_img:
            test_labels.append(1)
        else:
            test_labels.append(0)

        if count % 100 == 0:
            print(f"{count} images processed")
        count += 1
        
    print('total number of images processed:', count)

    # ------------------------------------------
    print('---------------------------------------------------')

    # convert to np arrays
    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)

    test_data = np.asarray(test_data)
    test_labels = np.asarray(test_labels)

    # Shuffle both train and test data
    train_data, train_labels = shuffle(train_data, train_labels, random_state=0)
    test_data, test_labels = shuffle(test_data, test_labels, random_state=0)

    np.save('train_data.npy', train_data)
    np.save('train_labels.npy', train_labels)
    np.save('test_data.npy', test_data)
    np.save('test_labels.npy', test_labels)

    # Note: Load the data like this -
    # train_data = np.load('train_data.npy')
    # train_labels = np.load('train_labels.npy')

    print('train data and test data saved to npy files in cwd')