#!/usr/bin/python
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.losses import *
from keras.preprocessing.image import *
from os.path import isfile
from tqdm import tqdm
import random
from glob import glob
import skimage.io as io
import skimage.transform as tr
import SimpleITK as sitk
from pushover import Client
import matplotlib.pyplot as plt

# img helper functions

def print_info(x):
    print(str(x.shape) + ' - Min: ' + str(x.min()) + ' - Mean: ' + str(x.mean()) + ' - Max: ' + str(x.max()))
    
def show_samples(x, y, num):
    two_d = True if len(x.shape) == 4 else False
    rnd = np.random.permutation(len(x))
    for i in range(0, num, 2):
        plt.figure(figsize=(15, 5))
        for j in range(2):
            plt.subplot(1,4,1+j*2)
            img = x[rnd[i+j], ..., 0] if two_d else x[rnd[i], 8+8*j, ..., 0]
            plt.imshow(img.astype('float32'))
            plt.subplot(1,4,2+j*2)
            if y[rnd[i]].shape[-1] == 1:
                img = y[rnd[i+j], ..., 0] if two_d else y[rnd[i], 8+8*j, ..., 0]
            else:
                img = y[rnd[i+j]] if two_d else y[rnd[i], 8+8*j]
            plt.imshow(img.astype('float32'))
        plt.show()

def shuffle(x, y):
    perm = np.random.permutation(len(x))
    x = x[perm]
    y = y[perm]
    return x, y

def split(x, y, tr_size):
    tr_size = int(len(x) * tr_size)
    x_tr = x[:tr_size]
    y_tr = y[:tr_size]
    x_te = x[tr_size:]
    y_te = y[tr_size:]
    return x_tr, y_tr, x_te, y_te

def augment(x, y, vert=False, hori=False, rot=False):
    if vert:
        tmp = np.flip(x, axis=1)
        x = np.concatenate((x, tmp))
        tmp = np.flip(y, axis=1)
        y = np.concatenate((y, tmp))
    if hori:
        tmp = np.flip(x, axis=2)
        x = np.concatenate((x, tmp))
        tmp = np.flip(y, axis=2)
        y = np.concatenate((y, tmp))
    if rot:
        tmp = np.rot90(x, axes=(1,2))
        x = np.concatenate((x, tmp))
        tmp = np.rot90(y, axes=(1,2))
        y = np.concatenate((y, tmp))
    return x, y

def resize_yx_3d(img, size=(24,224,224)):
    assert(len(img.shape) == 4)
    assert(size[0] == img.shape[0])
    img2 = np.zeros((size[0], size[1], size[2], img.shape[-1]))
    for i in range(size[0]):
        img2[i] = tr.resize(img[i], (size[1],size[2]), mode='constant', preserve_range=True)
    return img2
    #return tr.resize(img, (size[0], size[1], size[2], img.shape[-1]))

def reshape_z(img):
    if img.shape[0] == 41:
        img = img[::2] # 41 --> 21
        padding = np.zeros((3,img.shape[1],img.shape[2],img.shape[3]))
        img = np.concatenate((img, padding)) # 21 --> 24
    elif img.shape[0] == 26:
        img = img[1:-1]
    return img
    
def to_2d(x):
    assert len(x.shape) == 5 # Shape: (#, Z, Y, X, C)
    return np.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]))

def to_3d(imgs, z):
    assert len(imgs.shape) == 4 # Shape: (#, Y, X, C)
    return np.reshape(imgs, (imgs.shape[0] / z, z, imgs.shape[1], imgs.shape[2], imgs.shape[3]))

def crop_img(img, z, y, x):
    if z != 0:
        img = img[z:-z]
    if y != 0:
        img = img[:, y:-y]
    if x != 0:
        img = img[:, :, x:-x]
    return img

def resize_img(img, z, y, x):
    if z == -1:
        z = img.shape[0]
    img2 = np.zeros((size[0], size[1], size[2], img.shape[-1]))
    for i in range(img.shape[-1]):
        img2[..., i] = tr.resize(img[..., i], size, mode='constant')
    return tr.resize(img2[..., i], size, mode='constant')

def get_crop_area(img, threshold=0):
    y_arr = np.where(img.sum(axis=0) > threshold)[0]
    size = y_arr[-1] - y_arr[0] + 1
    y = y_arr[0]
    x_arr = np.where(img.sum(axis=0).sum(axis=0) > threshold)[0]
    x = (x_arr[0] + x_arr[-1]) // 2 - size // 2
    return y, x, size

def smart_crop(img, threshold=0):
    y_arr = np.where(img.sum(axis=0) > threshold)[0] # find height of object
    img = img[:, y_arr[0]:y_arr[-1]+1, ...] # crop in y
    x_arr = np.where(img.sum(axis=0).sum(axis=0) > threshold)[0] # find width of object
    x_center = (x_arr[0] + x_arr[-1]) // 2
    height = img.shape[1]
    fix = 0 if height % 2 == 0 else 1
    img = img[..., x_center-height//2:x_center+height//2+fix,:]
    return img

def n4_bias_correction(img):
    img = img[..., 0] if img.shape[-1] == 1 else img
    img = sitk.GetImageFromArray(img.astype('float32'))
    #img = sitk.Cast(img, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    img = sitk.N4BiasFieldCorrection(img, mask)
    return sitk.GetArrayFromImage(img)

# Label helper functions

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

def multilabel(img, channel):
    if channel == 1:
        img[img > 0.01] = 1
        img[img < 0.01] = 0
        return img
    else:
        step = img.max() // channel
        divider = img.max() * 0.99
        img2 = np.zeros((img.shape[0], img.shape[1], img.shape[2], channel))
        for c in range(channel):
            img2[img[..., 0] > divider, c] = 1
            img[img[..., 0] > divider, 0] = 0
            divider -= step
        return img2

def read_mhd_1(path, size=None, crop=None, normalize=False, label=0):
    img = io.imread(path, plugin='simpleitk')[..., np.newaxis]
    if label > 0:
        img = multilabel(img, label)
    if size and img.shape != size:
        img2 = np.zeros((size[0], size[1], size[2], img.shape[-1]))
        for i in range(img.shape[-1]):
            img2[..., i] = tr.resize(img[..., i], size, mode='constant')
        img = img2
    if crop:
        img = crop_img(img, crop[0], crop[1], crop[2])
    if normalize:
        img = (img-img.min()) / (img.max()-img.min())
    return img

def read_mhd(path, label=0, crop=None, reshape=False, size=None, bias=False, norm=False):
    # crop = (top, left, size)
    img = io.imread(path, plugin='simpleitk')[..., np.newaxis]
    img = multilabel(img, label) if label > 0 else img
    img = img[:, crop[0]:crop[0]+crop[2], crop[1]:crop[1]+crop[2]] if crop else img
    img = reshape_z(img) if reshape else img
    img = resize_yx_3d(img, size) if size else img
    img = n4_bias_correction(img) if bias else img
    img = (img-img.min()) / (img.max()-img.min()) if norm else img
    #img = (img-img.mean()) / img.std() if norm else img
    return img

def load_data(path, label=0, size=(24,224,224), bias=False, norm=False):
    files = glob(path)
    x, y = [], []
    for i in tqdm(range(len(files))):
        img = read_mhd(files[i])
        top, left, dim = get_crop_area(img)
        img = read_mhd(files[i], label=label, crop=(top, left, dim), reshape=True, size=size)
        y.append(img)
        files[i] = files[i].replace('/VOI_LABEL/', '/MHD/', 1)
        files[i] = files[i].replace('_LABEL.', '_ORIG.', 1)
        img = read_mhd(files[i], crop=(top, left, dim), reshape=True, size=size, bias=bias, norm=norm)
        x.append(img)
    x = np.array(x).astype('float16')
    y = np.array(y).astype('float16')
    return x, y

def load_data_seg(files, mode='ver', name='_', size=None, crop=None, label=1):
    files = glob(files)
    x, y = [], []
    for i in tqdm(range(len(files))):
        img = read_mhd_1(files[i], size, crop, False, label)
        y.append(img)
        files[i] = files[i].replace('/LABEL/', '/MHD/', 1)
        files[i] = files[i].replace('_LABEL.', '_ORIG.', 1)
        img = read_mhd_1(files[i], size, crop, True)
        x.append(img)
    x = np.array(x).astype('float16')
    y = np.array(y).astype('float16')
    x, y = shuffle(x, y)
    return x, y


# Models

def conv_bn(m, dim, acti, bn):
    m = Conv2D(dim, 3, activation=acti, padding='same')(m)
    return BatchNormalization()(m) if bn else m

def level_block(m, dim, depth, inc_rate, acti, dropout, bn, down, up):
    if depth > 0:
        n = conv_bn(m, dim, acti, bn)
        n = conv_bn(n, dim, acti, bn)
        if down == 'stride':
            m = Conv2D(dim, 3, strides=2, padding='same')(n) 
        else:
            m = MaxPooling2D()(n)
        m = level_block(m, int(inc_rate*dim), depth-1, inc_rate, acti, dropout, bn, down, up)
        if up == 'deconv':
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        else:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        m = Concatenate(axis=3)([n, m])
        m = conv_bn(m, dim, acti, bn)
    else:
        m = conv_bn(m, dim, acti, bn)
        n = Dropout(dropout)(n) if dropout else n
    return conv_bn(m, dim, acti, bn)

def UNet(img_shape, out_ch=1, start_ch=32, depth=4, inc_rate=1, activation='elu', dropout=0.5, bn=False, down='maxpool', up='upconv'):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, bn, down, up)
    o = Conv2D(out_ch, 1, activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)

def level_block_3d(m, dim, depth, factor, acti, dropout):
    if depth > 0:
        n = Conv3D(dim, 3, activation=acti, padding='same')(m)
        n = Dropout(dropout)(n) if dropout else n
        n = Conv3D(dim, 3, activation=acti, padding='same')(n)
        m = MaxPooling3D((1,2,2))(n)
        m = level_block_3d(m, int(factor*dim), depth-1, factor, acti, dropout)
        m = UpSampling3D((1,2,2))(m)
        m = Conv3D(dim, 2, activation=acti, padding='same')(m)
        m = Concatenate(axis=4)([n, m])
    m = Conv3D(dim, 3, activation=acti, padding='same')(m)
    return Conv3D(dim, 3, activation=acti, padding='same')(m)

def UNet_3D(img_shape, n_out=1, dim=8, depth=3, factor=1.5, acti='elu', dropout=None):
    i = Input(shape=img_shape)
    o = level_block_3d(i, dim, depth, factor, acti, dropout)
    o = Conv3D(n_out, 1, activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)

# Loss Functions

# 2TP / (2TP + FP + FN)
def f1(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

def f1_loss(y_true, y_pred):
    return 1-f1(y_true, y_pred)

dice = f1
dice_loss = f1_loss

def iou(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1. - intersection)

def iou_loss(y_true, y_pred):
    return -iou(y_true, y_pred)

def mae_img(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return mae(y_true_f, y_pred_f)

def bce_img(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return binary_crossentropy(y_true_f, y_pred_f)

def f1_bce(y_true, y_pred):
    return f1_loss(y_true, y_pred) + bce_img(y_true, y_pred)

# FP + FN
def error(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.sum(K.abs(y_true_f - y_pred_f)) / float(224*224)

# Notifications
    
def pushover(title, message):
    user = "u96ub3t5wu1nexmgi22xjs31jeb8y6"
    api = "avfytsyktracxood45myebobtry6yd"
    client = Client(user, api_token=api)
    client.send_message(message, title=title)
    
#from nipype.interfaces.ants import N4BiasFieldCorrection
#correct = N4BiasFieldCorrection()
#correct.inputs.input_image = in_file
#correct.inputs.output_image = out_file
#done = correct.run()
#img done.outputs.output_image