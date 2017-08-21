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
import skimage.morphology as mo
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
            plt.axis('off')
            plt.imshow(img.astype('float32'))
            plt.subplot(1,4,2+j*2)
            if y[rnd[i]].shape[-1] == 1:
                img = y[rnd[i+j], ..., 0] if two_d else y[rnd[i], 8+8*j, ..., 0]
            else:
                img = y[rnd[i+j]] if two_d else y[rnd[i], 8+8*j]
            plt.axis('off')
            plt.imshow(img.astype('float32'))
        plt.show()
        
def show_samples_2d(x, num, titles=None, axis_off=True, size=(20,20)):
    assert(len(x) >= 1)
    if titles:
        assert(len(titles) == len(x))
    rnd = np.random.permutation(len(x[0]))
    for row in range(num):
        plt.figure(figsize=size)
        for col in range(len(x)):
            plt.subplot(1,len(x), col+1)
            img = x[col][rnd[row], ..., 0] if x[col][rnd[row]].shape[-1] == 1 else x[col][rnd[row]]
            if axis_off:
                plt.axis('off')
            if titles:
                plt.title(titles[col])
            plt.imshow(img.astype('float32'), cmap='gray')
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

def augment(x, y, h_shift=[], v_flip=False, h_flip=False, rot90=False, edge_mode='minimum'):
    assert(len(x.shape) == 4)
    seg = False if len(y.shape) <= 2 else True
    if h_shift and h_shift != 0 and len(h_shift) != 0:
        tmp_x, tmp_y = [], []
        for shft in h_shift:
            if shft > 0:
                tmp = np.lib.pad(x[:, :, :-shft], ((0,0), (0,0), (shft,0), (0,0)), edge_mode)
                tmp_x.append(tmp)
                if seg:
                    tmp = np.lib.pad(y[:, :, :-shft], ((0,0), (0,0), (shft,0), (0,0)), edge_mode)
                else:
                    tmp = y
                tmp_y.append(tmp)
            else:
                tmp = np.lib.pad(x[:, :, -shft:], ((0,0), (0,0), (0,-shft), (0,0)), edge_mode)
                tmp_x.append(tmp)
                if seg:
                    tmp = np.lib.pad(y[:, :, -shft:], ((0,0), (0,0), (0,-shft), (0,0)), edge_mode)
                else:
                    tmp = y
                tmp_y.append(tmp)
        x = np.concatenate((x, np.concatenate(tmp_x)))
        y = np.concatenate((y, np.concatenate(tmp_y)))
    if v_flip:
        tmp = np.flip(x, axis=1)
        x = np.concatenate((x, tmp))
        if seg:
            tmp = np.flip(y, axis=1)
            y = np.concatenate((y, tmp))
        else:
            y = np.concatenate((y, y))
    if h_flip:
        tmp = np.flip(x, axis=2)
        x = np.concatenate((x, tmp))
        if seg:
            tmp = np.flip(y, axis=2)
            y = np.concatenate((y, tmp))
        else:
            y = np.concatenate((y, y))
    if rot90:
        tmp = np.rot90(x, axes=(1,2))
        x = np.concatenate((x, tmp))
        if seg:
            tmp = np.rot90(y, axes=(1,2))
            y = np.concatenate((y, tmp))
        else:
            y = np.concatenate((y, y))
    return x, y

def resize_3d(img, size):
    img2 = np.zeros((img.shape[0], size[0], size[1], img.shape[-1]))
    for i in range(img.shape[0]):
        img2[i] = tr.resize(img[i], (size[0], size[1]), mode='constant', preserve_range=True)
    return img2

def to_2d(x):
    assert len(x.shape) == 5 # Shape: (#, Z, Y, X, C)
    return np.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]))

def to_3d(imgs, z):
    assert len(imgs.shape) == 4 # Shape: (#, Y, X, C)
    return np.reshape(imgs, (imgs.shape[0] / z, z, imgs.shape[1], imgs.shape[2], imgs.shape[3]))

def get_crop_area(img, threshold=0):
    y_arr = np.where(img.sum(axis=0) > threshold)[0]
    size = y_arr[-1] - y_arr[0] + 1
    y = y_arr[0]
    x_arr = np.where(img.sum(axis=0).sum(axis=0) > threshold)[0]
    x = (x_arr[0] + x_arr[-1]) // 2 - size // 2
    return y, x, size

def n4_bias_correction(img):
    img = sitk.GetImageFromArray(img[..., 0].astype('float32'))
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    img = sitk.N4BiasFieldCorrection(img, mask)
    return sitk.GetArrayFromImage(img)[..., np.newaxis]

def handle_specials(img):
    if img.shape[0] == 26:
        img = img[1:-1]
    elif img.shape[0] == 20:
        img = np.lib.pad(img, ((2,2), (0,0), (0,0), (0,0)), 'minimum')
    return img

def erode(imgs, amount=3):
    imgs = imgs.sum(axis=-1)
    for i in range(len(imgs)):
        imgs[i] = mo.erosion(imgs[i], mo.square(amount))
    return imgs[..., np.newaxis]

def add_noise(imgs, amount=3):
    imgs = imgs.sum(axis=-1)
    for i in range(len(imgs)):
        if i % 2 == 0:
            imgs[i] = mo.dilation(imgs[i], mo.square(amount))
        else:
            imgs[i] = mo.erosion(imgs[i], mo.square(amount))
    return imgs[..., np.newaxis]
            

# Label helper functions

def to_classes(y, start, end, step=1):
    age_range = end - start
    num_classes = int(round(age_range / step))
    labels = np.zeros((len(y), num_classes))
    idx = (y - start) / step
    for i in range(len(idx)):
        labels[i, int(idx[i])] = 1
    return labels

def y_center(img, smooth=20, crop=100):
    # Get Sum of y-axis values
    y = img.sum(axis=-1).sum(axis=-1).sum(axis=0)
    # Smooth the values and apply the crop region
    y_vec = np.convolve(y, np.ones(smooth)/smooth, mode='same')[crop:-crop]
    # 2nd derivative of min will be max - get its index
    return np.gradient(np.gradient(y_vec)).argmax() + crop

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

def normalize(x, mean, std):
    return (x - x.mean()) / x.std()

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

def read_mhd(path, label=0, crop=None, size=None, bias=False, norm=False):
    img = io.imread(path, plugin='simpleitk')[..., np.newaxis].astype('float64')
    img = handle_specials(img)
    img = multilabel(img, label) if label > 0 else img
    img = img[:, crop[0]:crop[0]+crop[2], crop[1]:crop[1]+crop[2]] if crop else img
    #img = img[:, crop[0]:-2*crop[1]+crop[0], crop[1]:-1*crop[1]] if crop else img
    img = resize_3d(img, size) if size else img
    img = n4_bias_correction(img) if bias else img
    img = (img - img.mean()) / img.std() if norm else img
    return img.astype('float32')

def load_data(path, label=0, size=(24,224,224), bias=False, norm=False, to2d=False):
    files = glob(path)
    x, y = [], []
    for i in tqdm(range(len(files))):
        img = read_mhd(files[i])
        top, left, dim = get_crop_area(img)
        img = read_mhd(files[i], label=label, crop=(top, left, dim), size=size)
        if to2d:
            for layer in img:
                y.append(layer)
        else:
            y.append(img)
        files[i] = files[i].replace('/VOI_LABEL/', '/MHD/', 1)
        files[i] = files[i].replace('_LABEL.', '_ORIG.', 1)
        img = read_mhd(files[i], crop=(top, left, dim), size=size, bias=bias, norm=norm)
        if to2d:
            for layer in img:
                x.append(layer)
        else:
            x.append(img)
    x = np.array(x)
    y = np.array(y)
    return x, y

def load_data_age(files, size=None, crop=None, bias=False, norm=False, 
                  to2d=False, smart_crop=False):
    files = glob(files)
    x, y = [], []
    for i in tqdm(range(len(files))):
        if crop:
            if smart_crop:
                img = read_mhd(files[i])
                c = y_center(img)
                crop[0] = c - crop[2] // 2
        img = read_mhd(files[i], crop=crop, size=size, bias=bias, norm=norm)
        f = files[i].split('_')
        age = int(f[3]) + int(f[4]) / 12.
        if to2d:
            for layer in img:
                x.append(layer)
                y.append(age)
        else:
            x.append(img)
            y.append(age)
    x = np.array(x)
    y = np.array(y)
    return x, y

def print_weights(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape)) #try only "param"
    finally:
        f.close()

# Models

def conv_block(m, dim, acti, bn, res, do=0):
    n = Conv2D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Add()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Add()([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
    return m

def UNet(img_shape, out_ch=1, start_ch=32, depth=4, inc_rate=1., activation='elu', 
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
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

def f1_np(y_true, y_pred):
    return (2. * (y_true * y_pred).sum() + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def f1_loss(y_true, y_pred):
    return 1-f1(y_true, y_pred)

def f2(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (5. * intersection + 1.) / (4. * K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

def f2_loss(y_true, y_pred):
    return 1-f2(y_true, y_pred)

dice = f1
dice_loss = f1_loss

def iou(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1. - intersection)

def iou_np(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1.) / (y_true.sum() + y_pred.sum() + 1. - intersection)

def iou_loss(y_true, y_pred):
    return -iou(y_true, y_pred)

def precision(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.) / (K.sum(y_pred_f) + 1.)

def precision_np(y_true, y_pred):
    return ((y_true * y_pred).sum() + 1.) / (y_pred.sum() + 1.)

def recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.) / (K.sum(y_true_f) + 1.)

def recall_np(y_true, y_pred):
    return ((y_true * y_pred).sum() + 1.) / (y_true.sum() + 1.)

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

def error_np(y_true, y_pred):
    return (abs(y_true - y_pred)).sum() / float(len(y_true.flatten()))

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