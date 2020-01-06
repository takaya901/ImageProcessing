# %%
import numpy as np
import glob
import h5py

from keras.preprocessing.image import load_img, img_to_array

import h5py
import time

import matplotlib.pylab as plt
import matplotlib.pyplot as plot

import keras.backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
from keras.models import load_model

# datasetpath = output_path + '/dataset.hdf5'
patch_size = 32
batch_size = 256
epoch = 200
loss_list = []

def normalization(X):
    return X / 127.5 - 1

def load_data(datasetpath):
    with h5py.File(datasetpath, "r") as hf:
        X_full_train = hf["TrainWithoutTarget"][:].astype(np.float32)
        X_full_train = normalization(X_full_train)
        X_sketch_train = hf["TrainWithTarget"][:].astype(np.float32)
        X_sketch_train = normalization(X_sketch_train)
        X_full_val = hf["TestWithoutTarget"][:].astype(np.float32)
        X_full_val = normalization(X_full_val)
        X_sketch_val = hf["TestWithTarget"][:].astype(np.float32)
        X_sketch_val = normalization(X_sketch_val)
        return X_full_train, X_sketch_train, X_full_val, X_sketch_val

def conv_block_unet(x, f, name, bn_axis, bn=True, strides=(2,2)):
    x = LeakyReLU(0.2)(x)
    x = Conv2D(f, (3,3), strides=strides, name=name, padding='same')(x)
    if bn: x = BatchNormalization(axis=bn_axis)(x)
    return x

def up_conv_block_unet(x, x2, f, name, bn_axis, bn=True, dropout=False):
    x = Activation('relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(f, (3,3), name=name, padding='same')(x)
    if bn: x = BatchNormalization(axis=bn_axis)(x)
    if dropout: x = Dropout(0.5)(x)
    x = Concatenate(axis=bn_axis)([x, x2])
    return x

def generator_unet_upsampling(img_shape, disc_img_shape, model_name="generator_unet_upsampling"):
    filters_num = 64
    axis_num = -1
    channels_num = img_shape[-1]
    min_s = min(img_shape[:-1])

    unet_input = Input(shape=img_shape, name="unet_input")

    conv_num = int(np.floor(np.log(min_s)/np.log(2)))
    list_filters_num = [filters_num*min(8, (2**i)) for i in range(conv_num)]

    # Encoder
    first_conv = Conv2D(list_filters_num[0], (3,3), strides=(2,2), name='unet_conv2D_1', padding='same')(unet_input)
    list_encoder = [first_conv]
    for i, f in enumerate(list_filters_num[1:]):
        name = 'unet_conv2D_' + str(i+2)
        conv = conv_block_unet(list_encoder[-1], f, name, axis_num)
        list_encoder.append(conv)

    # prepare decoder filters
    list_filters_num = list_filters_num[:-2][::-1]
    if len(list_filters_num) < conv_num-1:
        list_filters_num.append(filters_num)

    # Decoder
    first_up_conv = up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                        list_filters_num[0], "unet_upconv2D_1", axis_num, dropout=True)
    list_decoder = [first_up_conv]
    for i, f in enumerate(list_filters_num[1:]):
        name = "unet_upconv2D_" + str(i+2)
        if i<2:
            d = True
        else:
            d = False
        up_conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i+3)], f, name, axis_num, dropout=d)
        list_decoder.append(up_conv)

    x = Activation('relu')(list_decoder[-1])
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(disc_img_shape[-1], (3,3), name="last_conv", padding='same')(x)
    x = Activation('tanh')(x)

    generator_unet = Model(inputs=[unet_input], outputs=[x])
    return generator_unet

def DCGAN_discriminator(img_shape, disc_img_shape, patch_num, model_name='DCGAN_discriminator'):
    disc_raw_img_shape = (disc_img_shape[0], disc_img_shape[1], img_shape[-1])
    list_input = [Input(shape=disc_img_shape, name='disc_input_'+str(i)) for i in range(patch_num)]
    list_raw_input = [Input(shape=disc_raw_img_shape, name='disc_raw_input_'+str(i)) for i in range(patch_num)]

    axis_num = -1
    filters_num = 64
    conv_num = int(np.floor(np.log(disc_img_shape[1])/np.log(2)))
    list_filters = [filters_num*min(8, (2**i)) for i in range(conv_num)]

    # First Conv
    generated_patch_input = Input(shape=disc_img_shape, name='discriminator_input')
    xg = Conv2D(list_filters[0], (3,3), strides=(2,2), name='disc_conv2d_1', padding='same')(generated_patch_input)
    xg = BatchNormalization(axis=axis_num)(xg)
    xg = LeakyReLU(0.2)(xg)

    # First Raw Conv
    raw_patch_input = Input(shape=disc_raw_img_shape, name='discriminator_raw_input')
    xr = Conv2D(list_filters[0], (3,3), strides=(2,2), name='raw_disc_conv2d_1', padding='same')(raw_patch_input)
    xr = BatchNormalization(axis=axis_num)(xr)
    xr = LeakyReLU(0.2)(xr)

    # Next Conv
    for i, f in enumerate(list_filters[1:]):
        name = 'disc_conv2d_' + str(i+2)
        x = Concatenate(axis=axis_num)([xg, xr])
        x = Conv2D(f, (3,3), strides=(2,2), name=name, padding='same')(x)
        x = BatchNormalization(axis=axis_num)(x)
        x = LeakyReLU(0.2)(x)

    x_flat = Flatten()(x)
    x = Dense(2, activation='softmax', name='disc_dense')(x_flat)

    PatchGAN = Model(inputs=[generated_patch_input, raw_patch_input], outputs=[x], name='PatchGAN')

    x = [PatchGAN([list_input[i], list_raw_input[i]]) for i in range(patch_num)]

    if len(x)>1:
        x = Concatenate(axis=axis_num)(x)
    else:
        x = x[0]

    x_out = Dense(2, activation='softmax', name='disc_output')(x)

    discriminator_model = Model(inputs=(list_input+list_raw_input), outputs=[x_out], name=model_name)

    return discriminator_model

def DCGAN(generator, discriminator, img_shape, patch_size):
    raw_input = Input(shape=img_shape, name='DCGAN_input')
    genarated_image = generator(raw_input)

    h, w = img_shape[:-1]
    ph, pw = patch_size, patch_size

    list_row_idx = [(i*ph, (i+1)*ph) for i in range(h//ph)]
    list_col_idx = [(i*pw, (i+1)*pw) for i in range(w//pw)]

    list_gen_patch = []
    list_raw_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            raw_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(raw_input)
            list_raw_patch.append(raw_patch)
            x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(genarated_image)
            list_gen_patch.append(x_patch)

    DCGAN_output = discriminator(list_gen_patch+list_raw_patch)

    DCGAN = Model(inputs=[raw_input],
                  outputs=[genarated_image, DCGAN_output],
                  name='DCGAN')

    return DCGAN

def load_generator(img_shape, disc_img_shape):
    model = generator_unet_upsampling(img_shape, disc_img_shape)
    return model

def load_DCGAN_discriminator(img_shape, disc_img_shape, patch_num):
    model = DCGAN_discriminator(img_shape, disc_img_shape, patch_num)
    return model

def load_DCGAN(generator, discriminator, img_shape, patch_size):
    model = DCGAN(generator, discriminator, img_shape, patch_size)
    return model

def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)

def inverse_normalization(X):
    return (X + 1.) / 2.

def to3d(X):
    if X.shape[-1]==3: return X
    b = X.transpose(3,1,2,0)
    c = np.array([b[0],b[0],b[0]])
    return c.transpose(3,1,2,0)

def plot_generated_batch(X_proc, X_raw, generator_model, batch_size, suffix):
    X_gen = generator_model.predict(X_raw)
    X_raw = inverse_normalization(X_raw)
    X_proc = inverse_normalization(X_proc)
    X_gen = inverse_normalization(X_gen)

    Xs = to3d(X_raw[:5])
    Xg = to3d(X_gen[:5])
    Xr = to3d(X_proc[:5])
    Xs = np.concatenate(Xs, axis=1)
    Xg = np.concatenate(Xg, axis=1)
    Xr = np.concatenate(Xr, axis=1)
    XX = np.concatenate((Xs,Xg,Xr), axis=0)

    plt.imshow(XX)
    plt.axis('off')
    plt.savefig(output_path + '/current_batch_'+suffix+'.png')
    plt.clf()
    plt.close()

def extract_patches(X, patch_size):
    list_X = []
    list_row_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[1] // patch_size)]
    list_col_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[2] // patch_size)]
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    return list_X

def get_disc_batch(procImage, rawImage, generator_model, batch_counter, patch_size):
    if batch_counter % 2 == 0:
        # produce an output
        X_disc = generator_model.predict(rawImage)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1
    else:
        X_disc = procImage
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)

    X_disc = extract_patches(X_disc, patch_size)
    return X_disc, y_disc


from PIL import Image
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
from keras.models import load_model
import tensorflow as tf
import glob

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

data_path = 'C:/Users/bumpo/Documents/Research/dataset/white/gray/test/'
# data_path = 'C:/Users/bumpo/Documents/Research/NDDS2/Captured/train/input/gray/'
gen_path = data_path + 'gen/'
# input_path = data_path + 'input/'
input_path = 'C:/Users/bumpo/Documents/Research/dataset/real_image/gray/input/'
gen_model = load_model(data_path + 'gen_model.h5')

input_imgs = glob.glob(input_path + '*.jpg')
img_num = len(input_imgs)
img_size = (128, 128)

for i in range(img_num):
    #load
    img = load_img(input_path + str(i) + '.jpg', target_size=img_size)
    array = img_to_array(img)
    in_array = []
    in_array.append(array)
    in_array = np.array(in_array)

    #generate
    start = time.time()
    norm = normalization(in_array)
    test = gen_model.predict(norm)
    test = inverse_normalization(test)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    #save
    Xg = to3d(test[:1])
    Xg = np.concatenate(Xg, axis=1)
    pred = array_to_img(Xg)
    save_img(gen_path + str(i) + '.jpg', pred)

# %%
