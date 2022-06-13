from audioop import avg
from skimage.color import rgb2gray
import os
import numpy as np
from lxml import etree
import glob
import numpy as np
from skimage import io
import csv
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
from pathlib import Path
import random
from tqdm import tqdm
from PIL import Image
import shutil
from skimage.util import img_as_ubyte

def xml_to_csv(path : Path = Path.cwd()):
    labels = os.listdir(path / 'train/labels')
    for lbl in labels:
        root = etree.parse(os.path.join('train/labels', lbl))
        objs = root.findall("object")
        bndboxes = np.zeros((len(objs), 5))
        for idx, obj in enumerate(objs):
            d = int(obj.find("difficult").text)
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            i = ymin
            j = xmin
            h = ymax - ymin
            w = xmax - xmin
            bndboxes[idx, :] = np.array([i, j, h, w, d])
        np.savetxt(os.path.join('labels_csv', lbl[:-4] + '.csv'), bndboxes, fmt='%i', delimiter=",")

def get_positives(path : Path = Path.cwd()):
    folder = path / "train/images/pos/"
    images = tqdm(folder.glob("*.jpg"), total = len(list(folder.glob("*.jpg"))))
    returned = []
    avg_height = []
    avg_width = []
    for img in images:
        images.set_description("Getting positive examples")
        file = path / "labels_csv" / f'{img.name.replace(".jpg","")}.csv' 
        if file.exists():
            csv_file = np.genfromtxt(file, delimiter=',')
            csv_file = csv_file.astype("int")
            if len(csv_file.shape) == 1 :
                csv_file = np.expand_dims(csv_file, axis = 0)
            num = 0
            for row in range(csv_file.shape[0]):
                i = csv_file[row,0]
                j = csv_file[row,1]
                h = csv_file[row,2]
                l = csv_file[row,3]
                d = csv_file[row,4]
                avg_height.append(h)
                avg_width.append(l)
                image = io.imread(img)
                exemple_pos = image[i:i+h+1,j:j+l+1,:]
                returned.append(exemple_pos)
                num = num + 1
    print(np.mean(np.array(avg_height)), np.mean(np.array(avg_width)))
    print(np.min(np.array(avg_height)), np.min(np.array(avg_width)))
    print(np.max(np.array(avg_height)), np.max(np.array(avg_width)))
    print(np.median(np.array(avg_height)), np.median(np.array(avg_width)))
    return(returned)

def generate_rotated_images(images_list):
    returned = []
    images = tqdm(images_list)
    for img in images:
        images.set_description("Generating rotated images")
        imageflipr = np.fliplr(img)
        imageflipud = np.flipud(img)
        returned.append(imageflipr)
        returned.append(imageflipud)
    return(returned)

def generate_cut_images(images_list, range : tuple = (2,3)):
    returned = []
    images = tqdm(images_list)
    for img in images:
        images.set_description("Generating cut images")
        h,w = img.shape[0], img.shape[1]
        a, b = range
        i = random.randint(0,2)
        j = random.randint(a,b)
        if i == 0:
            image = img[0:h//j,:,:]
            returned.append(image)
        if i == 1:
            image = img[:,0:w//j,:]
            returned.append(image)
        if i == 2:
            image = img[0:h//j,0:w//j,:]
            returned.append(image)
    return(returned)

def generate_negative_samples(images_list):
    returned = []
    images = tqdm(images_list)
    for img in images:
        images.set_description("Generating negative samples")
        h,w = img.shape[0], img.shape[1]
        x, y = random.randint(0,h-50), random.randint(0,w-50)
        i = random.randint(1,5)
        he = random.randint(10, h-x)
        if he//i < w-50:
            wi = he // i
        else:
            wi = w-y
        image = img[x:x+he,y:y+wi,:]
        returned.append(image)
    return(returned)

def get_negatives(path : Path = Path.cwd()):
    folder = path / "train/images/neg/"
    negatives = []
    returned = []
    images = tqdm(folder.glob("*.jpg"), total = len(list(folder.glob("*.jpg"))))
    for img in images:
        images.set_description("Getting negative examples")
        image = io.imread(img)
        negatives.append(image)
    returned += generate_negative_samples(negatives)
    returned += generate_negative_samples(negatives)
    returned += generate_negative_samples(negatives)
    returned += generate_negative_samples(negatives)
    returned += generate_negative_samples(negatives)
    returned += generate_negative_samples(negatives)
    returned += generate_negative_samples(negatives)
    returned += generate_negative_samples(negatives)
    returned += generate_negative_samples(negatives)
    return returned

def prepare_training_data(image_size : tuple = (50,50), path : Path = Path.cwd()):
    training_path = path / "training_data_processed"
    shutil.rmtree(training_path)
    pos_path = training_path / "pos"
    neg_path = training_path / "neg"
    pos_path.mkdir(parents = True, exist_ok=True)
    neg_path.mkdir(parents = True, exist_ok=True)
    positives = get_positives(path)
    rotated = generate_rotated_images(positives)
    total_positives = positives + rotated
    negatives = get_negatives(path)
    positives_imgs = tqdm(total_positives)
    negative_imgs = tqdm(negatives)
    id = 0
    for img in positives_imgs:
        positives_imgs.set_description("Saving positive training data")
        img = resize(img, image_size, anti_aliasing=True) 
        io.imsave(pos_path / f'{id}.jpg', img_as_ubyte(img), check_contrast=False)
        id += 1
    id = 0
    for img in negative_imgs:
        positives_imgs.set_description("Saving negative training data")
        img = resize(img, image_size, anti_aliasing=True) 
        io.imsave(neg_path / f'{id}.jpg', img_as_ubyte(img), check_contrast=False)
        id += 1
