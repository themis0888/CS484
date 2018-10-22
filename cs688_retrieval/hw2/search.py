"""
CUDA_VISIBLE_DEVICES=0 \
python -i search.py \
--data_path=/home/siit/navi/data/input_data/ukbench_small/ \
--meta_path=/home/siit/navi/data/meta_data/ukbench_small/ \
--model_name=vgg_16


CUDA_VISIBLE_DEVICES=1 python feature_extractor.py \
--data_path=/shared/data/danbooru2017/sample/0000/ \
--list_path=/shared/data/meta/danbooru2017/sample/0000/ \
--model_name=vgg_19
"""

import pdb
import random, time, os, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import skimage.io as skio
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/home/siit/navi/data/input_data/ukbench_small/')
parser.add_argument('--meta_path', type=str, dest='meta_path', default='/home/siit/navi/data/meta_data/ukbench_small/')
parser.add_argument('--save_path', type=str, dest='save_path', default='./output')
parser.add_argument('--model_path', type=str, dest='model_path', default='/home/siit/navi/data/models/')
parser.add_argument('--model_name', type=str, dest='model_name', default='vgg_19')
parser.add_argument('--loss', type=str, dest='loss', default='L2')

parser.add_argument('--memory_usage', type=float, dest='memory_usage', default=0.45)
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
		if config.loss == 'L1':
			dist_q_d = np.mean(np.abs(query_point - data_point))
		elif config.loss == 'L2':
			dist_q_d = np.sqrt(np.sum((query_point - data_point)**2))
		for k in reversed(range(top_k)):
			if dist_q_d < min_dist[k]:
				min_dist[k+1] = min_dist[k]
				min_dist[k] = dist_q_d
				min_data[k+1] = min_data[k]
				min_data[k] = data
			else: 
				break

	return min_data[:-1], min_dist[:-1]


def test():

	feature_dict = np.load(os.path.join(config.meta_path, 
		config.model_name + '_feature_fc7.npy')).item()

	query_list = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.data_path) 
		for f in filenames if 'query' in dp]

	query_filenames = [f for dp, dn, filenames in os.walk(config.data_path) 
		for f in filenames if 'query' in dp]

	data_list = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.data_path) 
		for f in filenames if ('test' in dp)] # and f not in query_filenames]

	"""
	sample_input = '/home/siit/navi/data/input_data/ukbench_small/test/image_00722.jpg'
	skio.imsave('sample.png', skio.imread('/home/siit/navi/data/input_data/ukbench_small/test/image_00722.jpg'))
	top_k_list = search_similar_image(sample_input, data_list, feature_dict)
	"""

	counter = 0
	num_correct = 0
	err_case = []
	for query in query_list:
		top_k_list, top_k_dist = search_similar_image(query, data_list, feature_dict)
		is_error = False
		for entry, dist in zip(top_k_list, top_k_dist):
			counter += 1
			if int(os.path.basename(query)[6:-4]) // 4 == int(os.path.basename(entry)[6:-4]) // 4:
				num_correct += 1
			else:
				# print(int(os.path.basename(query)[6:-4]) // 4, int(os.path.basename(entry)[6:-4]) // 4)
				is_error = True
		if is_error:	
			err_case.append([query, top_k_list, top_k_dist])

	print('\nSummary\n' +
		'Total number of images: {}\n'.format(counter) + 
		'Number of correct entries: {}\n'.format(num_correct) + 
		'Accuracy: {}\n'.format(num_correct/counter))

	return err_case

"""
def visualize(err_case, query_list):
def visualize(self, input_files, target_files, sample_path, counter, save_output = False, args = None):

	num_input = min(3, len(input_files))
	num_col = 3
	fig=plt.figure(figsize=(16, 16))

	output_files = self.sess.run(self.output_data, 
		feed_dict={self.X: input_files, self.Y: target_files, self.training: True})
	output_files = np.minimum(1, np.maximum(0, output_files))

	for i in range(num_input):

		input_file = input_files[i]
		output_file = output_files[i]
		target_file = target_files[i]
		#pdb.set_trace()

		fig.add_subplot(num_input, num_col, num_col*i+1)
		plt.imshow(input_file)
		fig.add_subplot(num_input, num_col, num_col*i+2)
		plt.imshow(output_file)
		fig.add_subplot(num_input, num_col, num_col*i+3)
		plt.imshow(target_file)

	plt.savefig(os.path.join(sample_path, '{0:06d}k_step.jpg'.format(counter)))
	plt.close()
	
	if save_output:
		skio.imsave(os.path.join(sample_path,'{:03d}.jpg'.format(counter)), output_file)

	return output_files
"""

start = time.time()
err_case = test()
end = time.time()
print('Time: {}sec'.format(end-start))


list_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.data_path) 
		for f in filenames]


