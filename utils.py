#!/usr/bin/python
import numpy as np
from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Activation, Dropout
from pushover import Client

def shuffle(x, y):
	perm = np.random.permutation(len(x))
	x = x[perm]
	y = y[perm]
	return x, y
	
def to_2d(x):
	assert len(x.shape) == 5 # Shape: (#, Z, Y, X, C)
	return np.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]))

def to_3d(imgs, z):
	assert len(imgs.shape) == 4 # Shape: (#, Y, X, C)
	return np.reshape(imgs, (imgs.shape[0] / z, z, imgs.shape[1], imgs.shape[2], imgs.shape[3]))

def lengthen(y, factor):
	arr = []
	for el in y:
		for i in range(factor):
			arr.append(el)
	return np.array(arr)

def shorten(y, factor):
	arr = []
	for i in range(0, len(y), factor):
		arr.append(y[i])
	return np.array(arr)

def level_block(m, dim, depth, inc_rate, activation, dropout):
	if depth > 0:
		n = Conv2D(dim, 3, activation=activation, padding='same')(m)
		n = Dropout(dropout)(n) if dropout else n
		n = Conv2D(dim, 3, activation=activation, padding='same')(n)
		m = MaxPooling2D()(n)
		m = level_block(m, int(inc_rate*dim), depth-1, inc_rate, activation, dropout)
		m = UpSampling2D()(m)
		m = Conv2D(dim, 2, activation=activation, padding='same')(m)
		m = Concatenate(axis=3)([n, m])
	m = Conv2D(dim, 3, activation=activation, padding='same')(m)
	m = Dropout(dropout)(m) if dropout else m
	return Conv2D(dim, 3, activation=activation, padding='same')(m)

def UNet(img_shape, n_out=1, init_dim=64, depth=4, inc_rate=2, activation='elu', dropout=None):
	i = Input(shape=img_shape)
	o = level_block(i, init_dim, depth, inc_rate, activation, dropout)
	o = Conv2D(n_out, (1, 1))(o)
	o = Activation('sigmoid')(o)
	return Model(inputs=i, outputs=o)
	
def pushover(title, message):
	user = "u96ub3t5wu1nexmgi22xjs31jeb8y6"
	api = "avfytsyktracxood45myebobtry6yd"
	client = Client(user, api_token=api)
	client.send_message(message, title=title)