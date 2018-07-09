from argparse import Namespace
from keras import backend as K
import numpy as np
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D

IS_KERAS2 = keras.__version__[0] == '2'
if IS_KERAS2:
  from fw_normalization import BatchNormOld

SR = 12000


def raw_vgg(args, input_length=12000 * 29, tf='melgram', normalize=None,
            decibel=False, last_layer=True, sr=None):
    ''' when length = 12000*29 and 512/256 dft/hop,
    melgram size: (n_mels, 1360)
    '''
    assert tf in ('stft', 'melgram')
    assert normalize in (None, False, 'no', 0, 0.0, 'batch', 'data_sample', 'time', 'freq', 'channel')
    assert isinstance(decibel, bool)

    if sr is None:
        sr = SR  # assumes 12000

    conv_until = args.conv_until
    trainable_kernel = args.trainable_kernel
    model = Sequential()
    # decode args
    fmin = args.fmin
    fmax = args.fmax
    if fmax == 0.0:
        fmax = sr / 2
    n_mels = args.n_mels
    trainable_fb = args.trainable_fb
    model.add(Melspectrogram(n_dft=512, n_hop=256, power_melgram=2.0,
                             input_shape=(1, input_length),
                             trainable_kernel=trainable_kernel,
                             trainable_fb=trainable_fb,
                             return_decibel_melgram=decibel,
                             sr=sr, n_mels=n_mels,
                             fmin=fmin, fmax=fmax,
                             name='melgram'))

    poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 4)]

    if normalize in ('batch', 'data_sample', 'time', 'freq', 'channel'):
        model.add(Normalization2D(normalize))
    model.add(get_convBNeluMPdrop(5, [32, 32, 32, 32, 32],
                                  [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
                                  poolings, model.output_shape[1:], conv_until=conv_until))
    if conv_until != 4:
        model.add(GlobalAveragePooling2D())
    else:
        model.add(Flatten())

    if last_layer:
        model.add(Dense(50, activation='sigmoid'))
    return model


def get_convBNeluMPdrop(num_conv_layers, nums_feat_maps,
                        conv_sizes, pool_sizes, input_shape,
                        conv_until=None):

    # [Convolutional Layers]
    model = Sequential(name='ConvBNEluDr')
    input_shape_specified = False

    if conv_until is None:
        conv_until = num_conv_layers  # end-inclusive.

    for conv_idx in xrange(num_conv_layers):
        # add conv layer
        if not input_shape_specified:
            model.add(Convolution2D(nums_feat_maps[conv_idx],
                                  conv_sizes[conv_idx][0], conv_sizes[conv_idx][1],
                                  input_shape=input_shape,
                                  border_mode='same',
                                  init='he_normal'))
            input_shape_specified = True
        else:
            model.add(Convolution2D(nums_feat_maps[conv_idx],
                                    conv_sizes[conv_idx][0], conv_sizes[conv_idx][1],
                                    border_mode='same',
                                    init='he_normal'))
        # add BN, Activation, pooling
        if IS_KERAS2:
            if K.image_data_format() == 'channels_first' or K.image_dim_ordering() == 'th':
                model.add(BatchNormOld(axis=1))
            elif K.image_data_format() == 'channels_last' or K.image_dim_ordering() == 'tf':
                model.add(BatchNormOld(axis=-1))
        else:
            model.add(BatchNormalization(axis=1, mode=2))
        model.add(keras.layers.advanced_activations.ELU(alpha=1.0))  # TODO: select activation

        model.add(MaxPooling2D(pool_size=pool_sizes[conv_idx]))
        if conv_idx == conv_until:
            break

    return model
def setup_model(mode, conv_until=None, compile=True,sr=None):
    assert mode in ('feature', 'tagger')
    if mode == 'feature':
        last_layer = False
    else:
        last_layer = True

    if conv_until is None:
        conv_until = 4

    K.set_image_dim_ordering('tf')

    args = Namespace(tf_type='melgram',  # which time-frequency to use
                     normalize='no', decibel=True, fmin=0.0, fmax=6000,  # mel-spectrogram params
                     n_mels=96, trainable_fb=False, trainable_kernel=False,  # mel-spectrogram params
                     conv_until=conv_until)  # how many conv layer to use? set it 4 if tagging.
    # set in [0, 1, 2, 3, 4] if feature extracting.

    # model = my_models.build_convnet_model(args=args, last_layer=last_layer)
    normalize = args.normalize
    if normalize in ('no', 'False'):
        normalize = None
    model = raw_vgg(args, tf=args.tf_type, normalize=normalize, decibel=args.decibel,
                    last_layer=last_layer, sr=sr)
    if compile:
        model.compile(optimizer=keras.optimizers.Adam(lr=5e-3),
                      loss='binary_crossentropy')

    return model
    # model.layers[1].save_weights('weights_layer{}_{}_new.hdf5'.format(conv_until, K._backend))
    # and use it!
    return model


if __name__ == '__main__':

    # load models that predict features from different level
    model = setup_model('feature')  # equal to setup_model('feature', 4), highest-level feature extraction




    # source for example
    src = np.load('1100103.clip.npy')  # (348000, )
    src = src[np.newaxis, :]  # (1, 348000)
    src = np.array([src]) # (1, 1, 348000) to make it batch

    #
    ph = tf.placeholder(tf.float32, shape=(1,1,348000))
    # feat = [md.predict(src)[0] for md in models] # get 5 features, each is 32-dim
    # feat = np.array(feat).reshape(-1) # (160, ) (flatten)
    # now use this feature for whatever MIR tasks.
    preds = model(ph)
    # feat = model.predict(src)[0]
    sess = tf.InteractiveSession()
    K.set_session(sess)
    sess.run( tf.global_variables_initializer())
    model.load_weights('weights_tf_dim.h5',by_name=True)
    feat = sess.run(preds, feed_dict={ph:src})[0]
    feat_gt = np.load('feats.npy')[:32]
    # assert all(np.isclose(feat_gt, feat))

    assert np.isclose(np.linalg.norm(feat),  np.linalg.norm(feat_gt)), 'exp: {} act: {}'.format(np.linalg.norm(feat_gt), np.linalg.norm(feat))
