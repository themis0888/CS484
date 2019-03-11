"""
python -i cityscape_preprocessor.py \
--data_path=/navi/data/input_data/ \
--data_format='png'
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/navi/data/input_data/mscoco/')
parser.add_argument('--meta_path', type=str, dest='meta_path', default='/navi/data/meta_data/mscoco/')
parser.add_argument('--sample_path', type=str, dest='output_path', default='./output')
parser.add_argument('--log_path', type=str, dest='log_path', default='./log')
parser.add_argument('--model_path', type=str, dest='model_path', default='/shared/data/models/')
parser.add_argument('--data_format', type=str, default='bin')
parser.add_argument('--dataset_name', type=str, default='CityScape')



config, unparsed = parser.parse_known_args() 

import os, random
import numpy as np
import skimage.measure as skms
import pdb
import skimage.io as skio
import scipy.misc
import pickle


# matrix resize function
def mat_resize(npy_file, fine_size, interp = 'bicubic'):
	img = npy_file
	img_layer = []
	for i in range(img.shape[-1]):
		img_layer.append(np.zeros(fine_size))
		img_layer[i] = scipy.misc.imresize(img[:,:,i], fine_size, interp = interp)
		img_layer[i] = np.expand_dims(img_layer[i], axis = -1)
		if i == 0:
			img_concat = img_layer[i]
		else:
			img_concat = np.concatenate((img_concat, img_layer[i]), axis = -1)
	return img_concat



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
output_path = os.path.join(data_path, config.dataset_name, config.data_format+ '_train_HR')
if not os.path.exists(output_path):
        os.makedirs(output_path)

output_path = os.path.join(data_path, config.dataset_name, config.data_format+ '_test_HR')
if not os.path.exists(output_path):
        os.makedirs(output_path)

for scale in range(2,5):
    output_path = os.path.join(data_path, config.dataset_name, 
        config.data_format+ '_train_LR_bicubic', 'X{}'.format(scale))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(data_path, config.dataset_name, 
        config.data_format+ '_test_LR_bicubic', 'X{}'.format(scale))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

#RGB value for the normalizing 
R_mean, G_mean, B_mean = 0, 0, 0
R_var, G_var, B_var = 0, 0, 0


# Processing 
# Save the original file and shringked files in pickle format 
# LR image has 4 dimension which contains RGB and Label of the object. 

def make_LR_files(label_file_list, mode):

    R_mean, G_mean, B_mean = 0, 0, 0
    R_var, G_var, B_var = 0, 0, 0

    print('Processing {} Set Start'.format(mode))
    num_files = len(label_file_list)
    counter = 0
    for label_path in train_label_files:

        label = skio.imread(label_path)
        label = np.expand_dims(label, axis = -1)

        file_name = os.path.basename(label_path).replace('_gtFine_labelIds.png', '')
        img_path = label_path.replace('/gtFine/', 
        '/leftImg8bit/').replace('gtFine_labelIds', 'leftImg8bit')
        img = skio.imread(img_path) /255

        R_mean, G_mean, B_mean = [R_mean, G_mean, B_mean] + np.mean(img, axis = (0,1))
        R_var, G_var, B_var = [R_var, G_var, B_var] + np.var(img, axis = (0,1))

        # Save the original HR file
        bin_path = os.path.join(data_path, config.dataset_name, config.data_format+ '_' + mode + '_HR', file_name)
        if config.data_format == 'bin':
            with open(bin_path, 'wb') as f:
                pickle.dump(img, f)
        else:
            skio.imsave(bin_path + '.png', img)

        # Save the shrinked files in 2, 3, 4 scale
        for scale in range(2,5):
            im_size = [int(img.shape[0]/scale), int(img.shape[1]/scale)]
            down_img = mat_resize(img, im_size) / 255
            
            down_lab = (mat_resize(label, im_size, interp = 'nearest') + 1) / 35
            temp = np.concatenate((down_img, down_lab), axis = -1)
            bin_path = os.path.join(data_path, config.dataset_name, 
            config.data_format+ '_' + mode + '_LR_bicubic', 'X{}'.format(scale), file_name)
            # pdb.set_trace()
            if config.data_format == 'bin':
                with open(bin_path, 'wb') as f:
                    pickle.dump(temp, f)
            else:
                skio.imsave(bin_path + '.png', temp)
        
        counter += 1
        if counter % 10 == 0:
            print('{:2.1f}% Done'.format(counter/num_files * 100))

    return R_mean, G_mean, B_mean, R_var, G_var, B_var


R_mean, G_mean, B_mean, R_var, G_var, B_var = make_LR_files(train_label_files, 'train')
R_mean, G_mean, B_mean, R_var, G_var, B_var = np.array([R_mean, G_mean, B_mean, R_var, G_var, B_var]) + make_LR_files(test_label_files, 'test')


num_files = len(test_label_files) + len(train_label_files)

R_mean, G_mean, B_mean = np.array([R_mean, G_mean, B_mean]) / num_files
R_var, G_var, B_var = np.array([R_var, G_var, B_var]) / num_files

print('RGB mean = {:1.5f}, {:1.5f}, {:1.5f}'.format(R_mean/255, G_mean/255, B_mean/255))
print('RGB var = {:1.5f}, {:1.5f}, {:1.5f}'.format(R_var, G_var, B_var))

