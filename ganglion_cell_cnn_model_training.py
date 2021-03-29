# This code tries to reproduce the results from the paper:
# McIntosh LT, Maheswaranathan N, Nayebi A, Ganguli S, Baccus SA. “Deep Learning Models
# of the Retinal Response to Natural Scenes”. Adv Neural Inf Process Syst. 2016

# Angel Canelo 2020.11.15

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

binned = pyret.spiketools.binspikes(spikes, time)
rate = pyret.spiketools.estfr(binned, time)
print('interpolated spikes', rate.shape)
# plt.plot(time[:500], rate[:500])
# plt.xlabel('Time (s)')
# plt.ylabel('Firing rate (Hz)')


filter_length_seconds = 0.5  # 500 ms filter
filter_length = int(filter_length_seconds / frame_rate)
sta, tax = pyret.filtertools.sta(time, stimulus, spikes, filter_length) #gives the spatial and temporal filters and the time axis
print('shape of spike triggered average is', sta.shape)
# fig, axes = pyret.visualizations.plot_sta(tax, sta)
# axes[0].set_title('Recovered spatial filter (STA)')
# axes[1].set_title('Recovered temporal filter (STA)')
# axes[1].set_xlabel('Time relative to spike (s)')
# axes[1].set_ylabel('Filter response')

# s_kernel, t_kernel = pyret.filtertools.decompose(sta)
# plt.plot(tax,t_kernel)

###################################
#cnn_model = models.Sequential()
inputs = keras.Input(shape=[20, 20, 1])
# injected noise strength
sigma = 0.1
n_cells = 1
# first layer
a = layers.Conv2D(16, 15, data_format="channels_last", kernel_regularizer=l2(1e-3), activation='relu', padding='same')(inputs)
a = layers.GaussianNoise(sigma)(a)
# second layer
a = layers.Conv2D(8, 9, data_format="channels_last", kernel_regularizer=l2(1e-3), activation='relu', padding='same')(a)
a = layers.GaussianNoise(sigma)(a)
a = layers.Flatten()(a)
outputs = layers.Dense(1, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3), activation='softplus')(a)
cnn_model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
lr = 1e-2; bz = 1000; nb_epochs = 50; val_split = 0.05

#x = Input(shape=stimulus.shape[1:])
x = stimulus.reshape(stimulus.shape[0], stimulus.shape[1], stimulus.shape[2], 1)
# Loss function and optimizer algorithm
cnn_model.compile(loss='poisson', optimizer=Adam(lr), metrics=['accuracy'])
base = os.path.abspath('.\\data\\')
os.makedirs(base, exist_ok=True)
# define model callbacks
cbs = [cb.ModelCheckpoint(os.path.join(base, 'weights-{epoch:03d}-{val_loss:.3f}.h5')),
       cb.TensorBoard(log_dir=base, histogram_freq=1, batch_size=5000, write_grads=True),
       cb.ReduceLROnPlateau(min_lr=0, factor=0.2, patience=10),
       cb.CSVLogger(os.path.join(base, 'training.csv')),
       cb.EarlyStopping(monitor='val_loss', patience=100)]
# train
history = cnn_model.fit(x, binned, batch_size=bz, epochs=nb_epochs,
                      callbacks=cbs, validation_split=val_split, shuffle=True)
cnn_model.save('cnn_model')
##################################################
plt.figure()
plt.plot(history.history['accuracy']); plt.plot(history.history['val_accuracy'])
plt.title('model accuracy'); plt.ylabel('accuracy'); plt.xlabel('epoch')
plt.legend(['train_accuracy', 'test_accuracy'], loc='upper right')
plt.figure()
plt.plot(history.history['loss']); plt.plot(history.history['val_loss'])
plt.title('model loss'); plt.ylabel('loss'); plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss'], loc='upper right')

predictions = cnn_model.predict(x)
predictions = predictions.reshape(predictions.shape[0])
pred_rate = pyret.spiketools.estfr(predictions, time)
# Save model
save_path = os.path.abspath('.\\data\\')
cnn_model.save(save_path, 'cnn_model.h5')

plt.figure()
plt.plot(time[:500], rate[:500], linewidth=5, color=(0.75,) * 3, alpha=0.7, label='True rate')
plt.plot(time[:500], pred_rate[:500], linewidth=2, color=(0.75, 0.1, 0.1), label='CNN predicted rate')
plt.legend(loc='upper right')
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (Hz)')
plt.show()