# Kapre version >0.0.2.3 (float32->floatx fixed version)
from argparse import Namespace
import models as my_models
from keras import backend as K
import keras
import numpy as np


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

    args = Namespace(tf_type='melgram',  # which time-frequency to use
                     normalize='no', decibel=True, fmin=0.0, fmax=6000,  # mel-spectrogram params
                     n_mels=96, trainable_fb=False, trainable_kernel=False,  # mel-spectrogram params
                     conv_until=conv_until)  # how many conv layer to use? set it 4 if tagging.
    # set in [0, 1, 2, 3, 4] if feature extracting.

    model = my_models.build_convnet_model(args=args, last_layer=last_layer)
    model.load_weights('weights_layer{}_{}.hdf5'.format(conv_until, K._backend),
                       by_name=True)
    # model.layers[1].save_weights('weights_layer{}_{}_new.hdf5'.format(conv_until, K._backend))
    # and use it!
    return model


if __name__ == '__main__':
    # main('tagger') # music tagger

    # load models that predict features from different level
    models = []
    model4 = setup_model('feature')  # equal to setup_model('feature', 4), highest-level feature extraction
    model3 = setup_model('feature', 3)  # low-level feature extraction.
    model2 = setup_model('feature', 2)  # lower-level..
    model1 = setup_model('feature', 1)  # lowerer...
    model0 = setup_model('feature', 0)  # lowererer.. no, lowest level feature extraction.

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
    # now use this feature for whatever MIR tasks.
    if keras.__version__[0] == '1':
      np.save('feats.npy', feat)
    else:
      feat_gt = np.load('feats.npy')
      assert all(np.isclose(feat_gt, feat))

      assert np.isclose(np.linalg.norm(feat),  9.916916)
