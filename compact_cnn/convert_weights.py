# Kapre version >0.0.2.3 (float32->floatx fixed version)
from argparse import Namespace
import models as my_models
from keras import backend as K
from keras.models import Model
from keras.layers import Input
import numpy as np
import tensorflow as tf


def setup_model(mode, conv_until=None):
    # setup stuff to build model

    # This is it. use melgram, up to 6000 (SR is assumed to be 12000, see model.py),
    # do decibel scaling
    assert mode in ('feature', 'tagger')
    if mode == 'feature':
        last_layer = False
    else:
        last_layer = True

    if conv_until is None:
        conv_until = 4

    K.set_image_dim_ordering('th')
    K.set_image_data_format('channels_first')

    args = Namespace(tf_type='melgram',  # which time-frequency to use
                     normalize='no', decibel=True, fmin=0.0, fmax=6000,  # mel-spectrogram params
                     n_mels=96, trainable_fb=False, trainable_kernel=False,  # mel-spectrogram params
                     conv_until=conv_until)  # how many conv layer to use? set it 4 if tagging.
    # set in [0, 1, 2, 3, 4] if feature extracting.

    model = my_models.build_convnet_model(args=args, last_layer=last_layer)
    model.load_weights('weights_layer{}_{}.hdf5'.format(conv_until, 'tensorflow'),
                     by_name=True)
    # model.layers[1].save_weights('weights_layer{}_{}_new.hdf5'.format(conv_until, K._backend))
    # and use it!
    return model


if __name__ == '__main__':
    # main('tagger') # music tagger

    # load models that predict features from different level
    model = setup_model('feature')  # equal to setup_model('feature', 4), highest-level feature extraction

    from keras.utils.conv_utils import convert_kernel
    ops = []
    for layer in model.layers[1].layers:
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
            original_w = K.get_value(layer.W)
            converted_w = convert_kernel(original_w)
            ops.append(tf.assign(layer.W, converted_w).op)
    K.get_session().run(ops)
    model.save_weights('my_weights_tensorflow.h5')


    # tf_model= tf.keras.estimator.model_to_estimator(keras_model=model4)

    # source for example
    src = np.load('1100103.clip.npy')  # (348000, )
    src = src[np.newaxis, :]  # (1, 348000)
    src = np.array([src]) # (1, 1, 348000) to make it batch

    #
    # feat = [md.predict(src)[0] for md in models] # get 5 features, each is 32-dim
    # feat = np.array(feat).reshape(-1) # (160, ) (flatten)
    # now use this feature for whatever MIR tasks.
