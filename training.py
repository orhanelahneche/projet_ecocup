from data_preparation import *
import numpy as np
from skimage import io
import csv
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
import glob
from sklearn import svm
from skimage.feature import hog
from skimage.color import rgb2gray
import pickle

def train_model(image_size : tuple = (50,50), path : Path = Path.cwd()):

    pos_path = path / "training_data_processed" / "pos"
    neg_path = path / "training_data_processed" / "neg"
    positives = tqdm(pos_path.glob("*.png"), total = len(list(pos_path.glob("*.png"))))
    negatives = tqdm(neg_path.glob("*.png"), total = len(list(neg_path.glob("*.png"))))

    total_training = len(list(pos_path.glob("*.png"))) + len(list(neg_path.glob("*.png")))
    X_train = np.zeros((total_training,1296))
    y_train = np.full((total_training), "")
    i = 0
    for pos in positives:
        positives.set_description("Gathering positive training data")
        image = io.imread(pos)
        im = hog(rgb2gray(image))
        X_train[i,:] = im
        y_train[i] = "E"
        i += 1

    for neg in negatives:
        negatives.set_description("Gathering negative training data")
        image = io.imread(neg)
        im = hog(rgb2gray(image))
        X_train[i,:] = im
        y_train[i] = "P"
        i += 1
    
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    filename = f'model{image_size}.sav'
    pickle.dump(clf, open(filename, 'wb'))
    
    return clf

def retrain_model(image_size : tuple = (50,50), path : Path = Path.cwd()):
    pos_path = path / "training_data_processed" / "pos"
    neg_path = path / "training_data_processed" / "neg"
    false_pos_path = path / "train" / "images" / "neg" / "false_pos"
    positives = tqdm(pos_path.glob("*.png"), total = len(list(pos_path.glob("*.png"))))
    negatives = tqdm(neg_path.glob("*.png"), total = len(list(neg_path.glob("*.png"))))
    negatives2 = tqdm(false_pos_path.glob("*.png"), total = len(list(false_pos_path.glob("*.png"))))
    total_training = len(list(pos_path.glob("*.png"))) + len(list(neg_path.glob("*.png"))) + len(list(false_pos_path.glob("*.png")))
    X_train = np.zeros((total_training,1296))
    y_train = np.full((total_training), "")
    i = 0
    for pos in positives:
        positives.set_description("Gathering positive training data")
        image = io.imread(pos)
        im = hog(rgb2gray(image))
        X_train[i,:] = im
        y_train[i] = "E"
        i += 1

    for neg in negatives:
        negatives.set_description("Gathering negative training data")
        image = io.imread(neg)
        im = hog(rgb2gray(image))
        X_train[i,:] = im
        y_train[i] = "P"
        i += 1

    for neg in negatives2:
        negatives.set_description("Gathering negative training data")
        image = io.imread(neg)
        im = hog(rgb2gray(image))
        X_train[i,:] = im
        y_train[i] = "P"
        i += 1
    
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    filename = f'model_retrained{image_size}.sav'
    pickle.dump(clf, open(filename, 'wb'))
    
    return clf