# This code tries to reproduce the results from the paper:
# McIntosh LT, Maheswaranathan N, Nayebi A, Ganguli S, Baccus SA. “Deep Learning Models
# of the Retinal Response to Natural Scenes”. Adv Neural Inf Process Syst. 2016

# Angel Canelo 2020.11.25

###### import ######################
import pyret
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
#from tensorflow import keras
from keras import datasets, layers, models
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.regularizers import l1, l2
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input
import keras.callbacks as cb
####################################
GPU_OPTIONS = tf.compat.v1.GPUOptions(allow_growth=True)
CONFIG = tf.compat.v1.ConfigProto(gpu_options=GPU_OPTIONS)
sess = tf.compat.v1.Session(config = CONFIG)


file_n = os.path.abspath('.\\data\\tutorial-data.h5')
data_file = h5py.File(file_n, 'r')
spikes = data_file['spike-times']  # Spike times for one cell
stimulus = data_file['stimulus'][()].astype(np.float64)
frame_rate = data_file['stimulus'].attrs.get('frame-rate')
print('shape of time is', stimulus.shape[0])
print('shape of spikes is', spikes.shape)
print('frame_rate = ', frame_rate)

stimulus -= stimulus.mean()
stimulus /= stimulus.std()
time = np.arange(stimulus.shape[0]) * frame_rate
x = stimulus.reshape(stimulus.shape[0], stimulus.shape[1], stimulus.shape[2], 1)

cnn_model = tf.keras.models.load_model('cnn_model')
cnn_model.summary()

#######################################################################################################
# Getting weights and plotting the filters, without average

# retrieve weights from the second hidden layer
filters, biases = cnn_model.layers[1].get_weights()
print(filters.shape)
# normalize filter values to 0-1 so we can visualize them
# filters = np.mean(filters, axis = (2))
# #f_min, f_max = filters.min(), filters.max()
# #filters = (filters - f_min) / (f_max - f_min)
# filters -= np.mean(filters)
# maxval = np.max(np.abs(filters))
# n_filters = filters.shape[2]
# plt.figure(); plt.clf()
# for i in range(n_filters):
# 	# get the filter
# 	f = filters[:, :, i]
# 	# plot each channel separately
# 	ax = plt.subplot(4,4, i+1)
# 	ax.set_xticks([])
# 	ax.set_yticks([])
# 	# plot filter channel in grayscale
# 	plt.imshow(f[:, :], cmap='seismic_r', interpolation='nearest',
#         aspect='equal', vmin=-maxval, vmax=maxval)
######################################################################################################

######################################################################################################
# Response weighted average, averaged feature maps (averaged filter or receptive field)
predictions = cnn_model.predict(x)
predictions = predictions.reshape(predictions.shape[0])
pred_rate = pyret.spiketools.estfr(predictions, time)
mod_model_layer1 = keras.Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[1].output)
mod_model_layer1.summary()
mod_model_layer2 = keras.Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[3].output)
mod_model_layer2.summary()
feature_maps = mod_model_layer1.predict(x)
avg_response = mod_model_layer1.predict(x)
print(feature_maps.shape)
feature_maps_layer2 = mod_model_layer2.predict(x)
avg_response_layer2 = mod_model_layer2.predict(x)
print(avg_response_layer2.shape)

avg_response = np.mean(avg_response, axis = (0))
#avg_response = np.average(avg_response, weights=pred_rate, axis = (0))
print(avg_response.shape)
avg_response -= np.mean(avg_response)
maxval = np.max(np.abs(avg_response))

avg_response_layer2 = np.mean(avg_response_layer2, axis = (0))
#avg_response_layer2 = np.average(avg_response_layer2, weights=pred_rate, axis = (0))
print(avg_response_layer2.shape)
avg_response_layer2 -= np.mean(avg_response_layer2)
maxval_2 = np.max(np.abs(avg_response_layer2))

# Plotting the stimulus
plt.figure(); plt.clf()
show_stim = np.squeeze(x)
plt.imshow(show_stim[500, :, :], cmap='gray')

plt.figure(); plt.clf()
ix = 1
for _ in range(feature_maps.shape[3]):
	# specify subplot and turn of axis
	ax = plt.subplot(4, 4, ix)
	ax.set_xticks([])
	ax.set_yticks([])
	# plot filter channel in grayscale
	plt.imshow(avg_response[ :, :, ix-1], cmap='seismic', interpolation='nearest',
        aspect='equal', vmin=-maxval, vmax=maxval)	# First dimension selects the frame to show
	ix += 1

plt.figure(); plt.clf()
ix = 1
for _ in range(avg_response_layer2.shape[2]):
	# specify subplot and turn of axis
	ax = plt.subplot(2, 4, ix)
	ax.set_xticks([])
	ax.set_yticks([])
	# plot filter channel in grayscale
	plt.imshow(avg_response_layer2[ :, :, ix-1], cmap='seismic', interpolation='nearest',
        aspect='equal', vmin=-maxval_2, vmax=maxval_2)	# First dimension selects the frame to show
	ix += 1
######################################################################################################

######################################################################################################
# Feature maps and activations for a specific frame
ix = 1
plt.figure(); plt.clf()
for _ in range(feature_maps.shape[3]):
	# specify subplot and turn of axis
	ax = plt.subplot(4, 4, ix)
	ax.set_xticks([])
	ax.set_yticks([])
	# plot filter channel in grayscale
	plt.imshow(feature_maps[10000, :, :, ix-1], cmap='seismic')	# First dimension selects the frame to show
	ix += 1

ix = 1
plt.figure(); plt.clf()
for _ in range(feature_maps_layer2.shape[3]):
	# specify subplot and turn of axis
	ax = plt.subplot(2, 4, ix)
	ax.set_xticks([])
	ax.set_yticks([])
	# plot filter channel in grayscale
	plt.imshow(feature_maps_layer2[10000, :, :, ix-1], cmap='seismic')	# First dimension selects the frame to show
	ix += 1
# Plotting the spatial averaged response of each subunit
avg_space = np.mean(feature_maps, axis = (1,2,3))
avg_space_single = np.mean(feature_maps, axis = (1,2))
print(avg_space.shape)
print(avg_space_single.shape)
plt.figure(); plt.clf()
plt.plot(time[:500], avg_space[:500], linewidth=2, color=(0.75, 0.1, 0.1))
for i in range(feature_maps.shape[3]):
	plt.plot(time[:500], avg_space_single[:500, i], color=(0.1, 0.75, 0.1), alpha=0.03*i)

avg_space_layer2 = np.mean(feature_maps_layer2, axis = (1,2,3))
avg_space_single_layer2 = np.mean(feature_maps_layer2, axis = (1,2))
plt.figure(); plt.clf()
plt.plot(time[:500], avg_space_layer2[:500], linewidth=2, color=(0.75, 0.1, 0.1))
for i in range(feature_maps_layer2.shape[3]):
	plt.plot(time[:500], avg_space_single_layer2[:500, i], color=(0.1, 0.75, 0.1), alpha=0.03*i)
plt.show()
###############################################################################################################
