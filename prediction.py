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
from sklearn.preprocessing import MinMaxScaler
import cv2
import random

class HeatMap():
    
    def __init__(self, original_image):
        self.image = np.zeros(original_image.shape[:2])
        
    def add_heat(self, coords):
        i = coords[0]
        j = coords[1]
        h = coords[2]
        w = coords[3]
        self.image[i:i+h+1,j:j+w+1] = self.image[i:i+h+1,j:j+w+1] + 30
        
    def remove_heat(self, coords):
        i = coords[0]
        j = coords[1]
        h = coords[2]
        w = coords[3]
        
        self.image[i:i+h+1,j:j+w+1] = self.image[i:i+h+1,j:j+w+1] - 30
        
    def heat_zones(self):
        scaler = MinMaxScaler()
        
        self.image = scaler.fit_transform(self.image)
        
        self.image = np.asarray(self.image * 255).astype(int)
        
        self.image = cv2.inRange(self.image,200,255)
        
        return self.image
    
    def heat_boxes(self):
        results = []
        heat_image = self.heat_zones()
        plt.imsave(os.path.join('first_try/','heat_map.jpg'), heat_image)
        contours, hierarchy = cv2.findContours(self.image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        idx = 0
        for cnt in contours:
            j,i,w,h = cv2.boundingRect(cnt)
            results.append([i,j,h,w])

        return results

def load_model(model : Path):
    loaded_model = pickle.load(open(model, 'rb'))
    return loaded_model


def detect_false_negs(clf, neg_path : Path):
    print()
    negatives = tqdm(neg_path.glob("*.jpg"), total = len(list(neg_path.glob("*.png"))))
    false_pos_save = neg_path / "false_pos"
    shutil.rmtree(false_pos_save, ignore_errors=True)
    false_pos_save.mkdir(parents=True, exist_ok=True)
    id = 0
    for neg in negatives:
        image = io.imread(neg)
        img = rgb2gray(image)
        for size in range(51, 152, 100):
                img_to_process = resize(img, (size,size), anti_aliasing=True)
                try:
                    img_to_process = img_as_ubyte(img_to_process)
                except:
                    pass
                for h1 in range(0,size-50, 12):
                    for w1 in range(0,size-50 , 12):
                        h2 = h1 + 50
                        w2 = w1 + 50
                        window = img_to_process[h1:h2,w1:w2]
                        I = hog(window)
                        predict = clf.predict(I.reshape(1, -1))
                        if (predict[0] == 'E'):
                            io.imsave(false_pos_save / f'{id}.png', window, check_contrast=False)
                            id += 1
    return


def sliding_window(image,clf):
    img = rgb2gray(image)
    img_copy = image
    height, width = img.shape[0], img.shape[1]
    img_to_process = resize(img, (250,250), anti_aliasing=True)
    try:
        img_to_process = img_as_ubyte(img_to_process)
    except:
        pass
    #cv2.imshow("aze", img_to_process)
    #cv2.waitKey(0)
    pos_result = []
    heat_map = HeatMap(img_copy)
    id = 0
    id2 = 0
    w = 50
    h = 100
    for size in range(150, 351, 100):
            img_to_process = resize(img, (size,size), anti_aliasing=True)
            try:
                img_to_process = img_as_ubyte(img_to_process)
            except:
                pass
            hratio = height / size
            wratio = width / size
            for h1 in range(0,size-50, 12):
                for w1 in range(0,size-50 , 12):
                    h2 = h1 + 50
                    w2 = w1 + 50
                    window = img_to_process[h1:h2,w1:w2]
                    I = hog(window)
                    predict = clf.predict(I.reshape(1, -1))
                    if (predict[0] == 'E'):
                        i, j, h, w = round(h1 * hratio), round(w1 * wratio), round(50 * hratio), round(50 * wratio)
                        heat_map.add_heat([i,j,h,w])
                        id = id + 1
                        pos_result.append([i,j,h,w])
    pos_result = heat_map.heat_boxes()
    for pos in pos_result:
        i = pos[0]
        j = pos[1]
        h = pos[2]
        w = pos[3]
        image = cv2.rectangle(image, (i,j), (i+h,j+w), (255), 2)
    plt.imsave(os.path.join(f'first_try/{random.randint(0,500)}.jpg'), image)
    id2+=1
    return pos_result
