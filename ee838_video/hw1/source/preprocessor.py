"""
python -i preprocessor.py \
--data_path=/navi/data/input_data/ \
"""
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/navi/data/input_data/mscoco/')
parser.add_argument('--meta_path', type=str, dest='meta_path', default='/navi/data/meta_data/mscoco/')
parser.add_argument('--sample_path', type=str, dest='output_path', default='./output')
parser.add_argument('--log_path', type=str, dest='log_path', default='./log')
parser.add_argument('--model_path', type=str, dest='model_path', default='/shared/data/models/')

config, unparsed = parser.parse_known_args() 

import os, random
import numpy as np
from model import *
import skimage.measure as skms
import data_loader
import pdb
import skimage.io as skio
import scipy.misc
import pickle


if not os.path.exists(config.output_path):
	os.mkdir(config.output_path)
if not os.path.exists(config.log_path):
	os.mkdir(config.log_path)

data_path = config.data_path

# List of string
# Path list of all files  
train_label_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(data_path)
		for f in filenames if 'gtFine' in dp and 'train' in dp and 'labelIds.png' in f]

train_img_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(data_path)
		for f in filenames if 'left' in dp and 'train' in dp and '.png' in f]

test_label_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.data_path)
		for f in filenames if 'gtFine' in dp and 'test' in dp and 'labelIds.png' in f]

test_img_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(data_path)
		for f in filenames if 'left' in dp and 'test' in dp and '.png' in f]


# Make the output directory 
output_path = os.path.join(data_path, 'CityScape', 'CityScape_train_HR')
if not os.path.exists(output_path):
        os.makedirs(output_path)

output_path = os.path.join(data_path, 'CityScape', 'CityScape_test_HR')
if not os.path.exists(output_path):
        os.makedirs(output_path)

for scale in range(2,5):
    output_path = os.path.join(data_path, 'CityScape', 
        'CityScape_train_LR_bicubic', 'X{}'.format(scale))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(data_path, 'CityScape', 
        'CityScape_test_LR_bicubic', 'X{}'.format(scale))
    if not os.path.exists(output_path):
        os.makedirs(output_path)


counter = 0
num_files = len(train_label_files)

# Processing 
# Save the original file and shringked files in pickle format 
# LR image has 4 dimension which contains RGB and Label of the object. 
for label_path in train_label_files:

    label = skio.imread(label_path)
    label = np.expand_dims(label, axis = -1)

    file_name = os.path.basename(label_path).replace('_gtFine_labelIds.png', '')
    img_path = label_path.replace('/gtFine_trainvaltest/gtFine/', 
    '/leftImg8bit/').replace('gtFine_labelIds', 'leftImg8bit')
    img = skio.imread(img_path)

    # Save the original HR file
    bin_path = os.path.join(data_path, 'CityScape', 'CityScape_train_HR')
    with open(bin_path, 'wb') as f:
        pickle.dump(img, f)

    # Save the shrinked files in 2, 3, 4 scale
    for scale in range(2,5):
        im_size = [int(img.shape[0]/scale), int(img.shape[1]/scale)]
        down_img = mat_resize(img, im_size)
        
        down_lab = (mat_resize(label, im_size) + 1) / 35
        temp = np.concatenate((down_img, down_lab), axis = -1)
        bin_path = os.path.join(data_path, 'CityScape', 
        'CityScape_train_LR_bicubic', 'X{}'.format(scale), file_name)
        with open(bin_path, 'wb') as f:
            pickle.dump(temp, f)
    
    counter += 1
    if counter % 10 == 0:
        print('{:2.2f} Done'.format(counter/num_files))



counter = 0
num_files = len(test_label_files)

# Processing 
# Repeat for the test files 
for label_path in test_label_files:

    label = skio.imread(label_path)
    label = np.expand_dims(label, axis = -1)

    file_name = os.path.basename(label_path).replace('_gtFine_labelIds.png', '')
    img_path = label_path.replace('/gtFine_trainvaltest/gtFine/', 
    '/leftImg8bit/').replace('gtFine_labelIds', 'leftImg8bit')
    img = skio.imread(img_path)

    # Save the original HR file
    bin_path = os.path.join(data_path, 'CityScape', 'CityScape_test_HR')
    with open(bin_path, 'wb') as f:
        pickle.dump(img, f)

    # Save the shrinked files in 2, 3, 4 scale
    for scale in range(2,5):
        im_size = [int(img.shape[0]/scale), int(img.shape[1]/scale)]
        down_img = mat_resize(img, im_size)
        
        down_lab = (mat_resize(label, im_size) + 1) / 35
        temp = np.concatenate((down_img, down_lab), axis = -1)
        bin_path = os.path.join(data_path, 'CityScape', 
        'CityScape_test_LR_bicubic', 'X{}'.format(scale), file_name)
        with open(bin_path, 'wb') as f:
            pickle.dump(temp, f)
    
    counter += 1
    if counter % 10 == 0:
        print('{:2.2f} Done'.format(counter/num_files))


    



def mat_resize(npy_file, fine_size):
	img = npy_file
	img_layer = []
	for i in range(img.shape[-1]):
		img_layer.append(np.zeros(fine_size))
		img_layer[i] = scipy.misc.imresize(img[:,:,i], fine_size)
		img_layer[i] = np.expand_dims(img_layer[i], axis = -1)
		if i == 0:
			img_concat = img_layer[i]
		else:
			img_concat = np.concatenate((img_concat, img_layer[i]), axis = -1)
	return img_concat