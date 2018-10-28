"""
CUDA_VISIBLE_DEVICES=0 python -i mnist_classification.py \
#--data_path=/shared/data/mnist_png
"""
import tensorflow as tf
import os
import numpy as np
import skimage.io as skio
import scipy.misc
import OpenEXR, Imath
import pdb

VGG_MEAN = [103.939, 116.779, 123.68]
# extensions = ('.jpg', '.png')
# /shared/data/mnist_png/train/0/1.png

# fine_list: str, list, bool, bool -> list of str
# Find the file list recursively
def file_list(path, extensions, sort=True, path_label = False):
	if path_label == True:
		result = [(os.path.join(dp, f) + ' % ' + os.path.join(dp, f).split('/')[-2])
		for dp, dn, filenames in os.walk(path) 
		for f in filenames if os.path.splitext(f)[1] in extensions]
	else:
		result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) 
		for f in filenames if os.path.splitext(f)[1] in extensions]
	if sort:
		result.sort()

	return result



def make_list_file(path, save_path, extensions, path_label = False, iter = 1):
	# make the save dir if it is not exists
	#save_path = os.path.join(path, 'meta')
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	print('Finding all input files...')
	file_lst = file_list(path, extensions, True, path_label)
	lenth = len(file_lst)

	print('Writing input file list...')
	for itr in range(iter):
		# save the file inside of the meta/ folder
		f = open(os.path.join(save_path, 'path_label_list{0:03d}.txt'.format(itr)), 'w')
		for line in file_lst[int((itr)*lenth/iter):int((itr+1)*lenth/iter)]:
			f.write(line + '\n')
		f.close()

	print('Listing completed...')


def make_dict_file(path, save_path, label_list, extensions, path_label = False, iter = 1):
	# make the save dir if it is not exists
	#save_path = os.path.join(path, 'meta')
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	print('Finding all input files...')
	file_lst = file_list(path, extensions, True, path_label)
	lenth = len(file_lst)
	path_label_dict = {}
	print('Writing input file list...')
	for itr in range(iter):
		# save the file inside of the meta/ folder
		for i in range(lenth):
			path_label_dict[file_lst[i]] = label_list[i]
	np.save(os.path.join(save_path, 'path_label_dict.npy'), path_label_dict)

	print('Listing completed...')



# queue_data(lst, ['0', '1', '2'], norm=True, convert = 'rgb2gray')
# queue_data does not consider the batch size but return the all data on the list.
def queue_data_list(file_list, label_list, im_size = [28,28], label_processed = True, norm=True, convert = None):
	# Batch frame fit into the image size 
	batch_size = len(file_list)
	im_batch = np.zeros([batch_size] + im_size)
	input_labels = []
	gt_labels = []
	# Reading from the list
	for i in range(batch_size):
		impath, input_label = file_list[i].split(' % ')
		input_label.replace('\n', '')
		# return the index of the label 
		gt_labels.append(np.array(input_label))
		if not label_processed:
			input_labels.append(label_list.index(input_label))
		else:
			input_labels.append(np.array(input_label))
		img = np.asarray(skio.imread(impath))
		if img.ndim < 3:
			img = np.expand_dims(img, axis = -1)
			img = np.concatenate((img, img, img), axis = -1)
		img = mat_resize(img, im_size)
		im_batch[i] = img
	if norm == True : 
		im_batch -= np.ndarray(VGG_MEAN)
	if convert == 'rgb2gray':
		im_batch = np.mean(im_batch, axis=3)
	# Label processing 
	n_classes = len(label_list)
	if not label_processed:
		label_indices = np.array([input_labels]).reshape(-1)
		input_labels = np.eye(n_classes)[label_indices]
	return im_batch, input_labels, gt_labels


def queue_data_dict(file_list, im_size = [28, 28], norm=True, image_resize = True, crop_cord = None):
	# Batch frame fit into the image size 
	batch_size = len(file_list)
	imgs = []
	gt_labels = []
	# Reading from the list
	for i in range(batch_size):
		impath = file_list[i]
		# return the index of the label 
		#pdb.set_trace()
		img = np.asarray(skio.imread(impath))
		if img.ndim < 3:
			img = np.expand_dims(img, axis = -1)
			img = np.concatenate((img, img, img), axis = -1)
		if type(crop_cord) == np.ndarray:
			img = img[int((img.shape[0]-im_size[0])*crop_cord[0,i]):int((img.shape[0]-im_size[0])*crop_cord[0,i])+im_size[0],
				int((img.shape[1]-im_size[1])*crop_cord[1,i]):int((img.shape[1]-im_size[1])*crop_cord[1,i])+im_size[1]]
		if image_resize: 
			img = mat_resize(img, im_size)
		imgs.append(img)
	im_batch = np.zeros([batch_size] + list(imgs[0].shape))
	for i in range(batch_size):
		im_batch[i] = imgs[i]
	if norm == True : 
		im_batch -= np.array(VGG_MEAN, dtype=np.float32)

	return im_batch


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

import numpy as np
import scipy.misc
import OpenEXR, Imath

class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# Read and prepare 8-bit image in a specified resolution
def readLDR(file, sz, clip=True, sc=1.0):
    try:
        x_buffer = scipy.misc.imread(file)

        # Clip image, so that ratio is not changed by image resize
        if clip:
            sz_in = [float(x) for x in x_buffer.shape]
            sz_out = [float(x) for x in sz]

            r_in = sz_in[1]/sz_in[0]
            r_out = sz_out[1]/sz_out[0]

            if r_out / r_in > 1.0:
                sx = sz_in[1]
                sy = sx/r_out
            else:
                sy = sz_in[0]
                sx = sy*r_out

            yo = np.maximum(0.0, (sz_in[0]-sy)/2.0)
            xo = np.maximum(0.0, (sz_in[1]-sx)/2.0)

            x_buffer = x_buffer[int(yo):int(yo+sy),int(xo):int(xo+sx),:]

        # Image resize and conversion to float
        x_buffer = scipy.misc.imresize(x_buffer, sz)
        x_buffer = x_buffer.astype(np.float32)/255.0

        # Scaling and clipping
        if sc > 1.0:
            x_buffer = np.minimum(1.0, sc*x_buffer)

        x_buffer = x_buffer[np.newaxis,:,:,:]

        return x_buffer
            
    except Exception as e:
        raise IOException("Failed reading LDR image: %s"%e)

# Write exposure compensated 8-bit image
def writeLDR(img, file, exposure=0):
    
    # Convert exposure fstop in linear domain to scaling factor on display values
    sc = np.power(np.power(2.0, exposure), 0.5)

    try:
        scipy.misc.toimage(sc*np.squeeze(img), cmin=0.0, cmax=1.0).save(file)
    except Exception as e:
        raise IOException("Failed writing LDR image: %s"%e)

# Write HDR image using OpenEXR
def writeEXR(img, file):
    try:
        img = np.squeeze(img)
        sz = img.shape
        header = OpenEXR.Header(sz[1], sz[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(file, header)
        R = (img[:,:,0]).astype(np.float16).tostring()
        G = (img[:,:,1]).astype(np.float16).tostring()
        B = (img[:,:,2]).astype(np.float16).tostring()
        out.writePixels({'R' : R, 'G' : G, 'B' : B})
        out.close()
    except Exception as e:
        raise IOException("Failed writing EXR: %s"%e)


# Read training data (HDR ground truth and LDR JPEG images)
def load_training_pair(name_hdr, name_jpg):

    data = np.fromfile(name_hdr, dtype=np.float32)
    ss = len(data)
    
    if ss < 3:
        return (False,0,0)

    sz = np.floor(data[0:3]).astype(int)
    npix = sz[0]*sz[1]*sz[2]
    meta_length = ss - npix

    # Read binary HDR ground truth
    y = np.reshape(data[meta_length:meta_length+npix], (sz[0], sz[1], sz[2]))

    # Read JPEG LDR image
    x = scipy.misc.imread(name_jpg).astype(np.float32)/255.0

    return (True,x,y)