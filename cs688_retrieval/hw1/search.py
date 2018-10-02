"""
CUDA_VISIBLE_DEVICES=0 \
python -i 211_VGG_feature_extractor.py \
--data_path=/home/siit/navi/data/input_data/ukbench_small/ \
--save_path=/home/siit/navi/data/meta_data/ukbench_small/ \
--model_name=vgg_19


CUDA_VISIBLE_DEVICES=1 python feature_extractor.py \
--data_path=/shared/data/danbooru2017/sample/0000/ \
--list_path=/shared/data/meta/danbooru2017/sample/0000/ \
--model_name=vgg_19
"""

import pdb
import random, time, os, sys
import numpy as np
import scipy
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import scipy.io as sio
import skimage.io as skio

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/home/siit/navi/data/input_data/ukbench_small/')
parser.add_argument('--meta_path', type=str, dest='meta_path', default='/home/siit/navi/data/meta_data/ukbench_small/')
parser.add_argument('--save_path', type=str, dest='save_path', default='./output')
parser.add_argument('--input_list', type=str, dest='input_list', default='path_label_list.txt')
parser.add_argument('--model_path', type=str, dest='model_path', default='/home/siit/navi/data/models/')
parser.add_argument('--model_name', type=str, dest='model_name', default='vgg_19')

parser.add_argument('--memory_usage', type=float, dest='memory_usage', default=0.45)
parser.add_argument('--n_classes', type=int, dest='n_classes', default=50)
parser.add_argument('--max_iter', type=int, dest='max_iter', default=300000)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=1)
parser.add_argument('--train_display', type=int, dest='train_display', default=200)
parser.add_argument('--val_display', type=int, dest='val_display', default=1000)
parser.add_argument('--val_iter', type=int, dest='val_iter', default=100)
config, unparsed = parser.parse_known_args() 

if os.path.exists(config.save_path):
	os.makedirs(config.save_path)

# search_similar_image: str, list of str, dict of ndarray, int -> str
# img address, list of img adress, features, int -> img address
# Find the top k images with biggest similarity
def search_similar_image(query, data_list, features, top_k = 4):
    
    query_point = features[query]
    min_dist = [np.inf for i in range(top_k+1)]
    min_data = [None for i in range(top_k+1)]       
        
    for data in data_list:
        data_point = features[data]
        dist_q_d = np.mean(np.abs(query_point - data_point))
        for k in reversed(range(top_k)):
            if dist_q_d < min_dist[k]:
                min_dist[k+1] = min_dist[k]
                min_dist[k] = dist_q_d
                min_data[k+1] = min_data[k]
                min_data[k] = data
            else: 
                break

    return min_data[:-1]

def test():

    feature_dict = np.load(os.path.join(config.meta_path, 'vgg_19_feature_prediction.npy')).item()
    
    query_list = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.data_path) 
		for f in filenames if 'query' in dp]



    data_list = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.data_path) 
		for f in filenames if ('test' in dp) and f not in query_list]

    sample_input = '/home/siit/navi/data/input_data/ukbench_small/test/image_00722.jpg'
    skio.imsave('sample.png', skio.imread('/home/siit/navi/data/input_data/ukbench_small/test/image_00722.jpg'))
    top_k_list = search_similar_image(sample_input, data_list, feature_dict)

    counter = 1
    for img in top_k_list:
        counter += 1
        skio.imsave('top_{}.png'.format(counter), skio.imread(img))


list_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.data_path) 
		for f in filenames]


