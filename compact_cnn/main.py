# Kapre version >0.0.2.3 (float32->floatx fixed version)
from argparse import Namespace
import models as my_models
from keras import backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
# from keras.utils.layer_utils import convert_kernel

import tensorflow as tf
import numpy as np
import sys

use_new = len(sys.argv) > 1
# def convert_model_layers(model):
#   ops = []
#   for layer in model.layers:

#     print(layer.__class__.__name__)
#     if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
#       original_w = K.get_value(layer.W)
#       print(original_w.shape)
#       kernel=original_w
#       # print(kernel[0,1,0,0])
#       slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
#       no_flip = (slice(None, None), slice(None, None))
#       slices[-2:] = no_flip
#       print(slices)
#       converted_w = np.copy(kernel[slices])
#       # print(converted_w[0,1,0,0])
#       # converted_w = convert_kernel(original_w)
#       print(converted_w.shape)
#       ops.append(tf.assign(layer.W, converted_w).op)
#   input('type something to continue')
#   return ops


def main(mode, conv_until=None):
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

    # assert K.image_dim_ordering() == 'th', ('image_dim_ordering should be "th". ' +
    #                                         'open ~/.keras/keras.json to change it.')
    # TODO what are the issues with the library
    # dimension ordering is off and I think it might be kapre
    # it could be fixed by changing the kapre version maybe

    args = Namespace(tf_type='melgram',  # which time-frequency to use
                     normalize='no', decibel=True, fmin=0.0, fmax=6000,  # mel-spectrogram params
                     n_mels=96, trainable_fb=False, trainable_kernel=False,  # mel-spectrogram params
                     conv_until=conv_until)  # how many conv layer to use? set it 4 if tagging.
    # set in [0, 1, 2, 3, 4] if feature extracting.
    if use_new:
      # K.set_image_dim_ordering('tf')
      K.set_image_dim_ordering('th')
    else:
      K.set_image_dim_ordering('th')
    model = my_models.build_convnet_model(args=args, last_layer=last_layer)
    model.summary()
    model.layers[1].summary()
    print(K.image_dim_ordering())
    if use_new:
      # model.load_weights('weights_layer{}_{}_new.hdf5'.format(conv_until, K._backend))
      model.load_weights('weights_layer{}_{}.hdf5'.format(conv_until, K._backend),
                         by_name=True)
    else:
      K.set_image_dim_ordering('tf')
      model.load_weights('weights_layer{}_{}.hdf5'.format(conv_until, K._backend),
                         by_name=True)
      convert_all_kernels_in_model(model.layers[1])
      # ops = convert_model_layers(model.layers[1])
      # K.get_session().run(ops)

      model.save_weights('weights_layer{}_{}_new.hdf5'.format(conv_until, K._backend))

    # model.summary()
    # and use it!
    return model


if __name__ == '__main__':
    # main('tagger') # music tagger

    # load models that predict features from different level
    models = []
    model4 = main('feature')  # equal to main('feature', 4), highest-level feature extraction
    model3 = main('feature', 3)  # low-level feature extraction.
    model2 = main('feature', 2)  # lower-level..
    model1 = main('feature', 1)  # lowerer...
    model0 = main('feature', 0)  # lowererer.. no, lowest level feature extraction.

    # prepare the models
    models.append(model4)
    models.append(model3)
    models.append(model2)
    models.append(model1)
    models.append(model0)

    # source for example
    src = np.load('1100103.clip.npy')  # (348000, )
    src = src[np.newaxis, :]  # (1, 348000)
    src = np.array([src]) # (1, 1, 348000) to make it batch

    #
    feat = [md.predict(src)[0] for md in models] # get 5 features, each is 32-dim
    feat = np.array(feat).reshape(-1) # (160, ) (flatten)
    norm_feats=  np.linalg.norm(feat)
    exp_norm_feats = 9.9169
    assert np.isclose(norm_feats, exp_norm_feats), 'expected {} got {}'.format(exp_norm_feats, norm_feats)
    # now use this feature for whatever MIR tasks.
