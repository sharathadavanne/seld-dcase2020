#
# The SELDnet architecture
#

from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.models import load_model
import keras
keras.backend.set_image_data_format('channels_first')
from IPython import embed


def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
                                rnn_size, fnn_size, weights, doa_objective):
    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))

    # CNN
    spec_cnn = spec_start
    for i, convCnt in enumerate(pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)

    # RNN
    spec_rnn = Reshape((data_in[-2], -1))(spec_cnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)

    # FC - DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)

    doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    doa = Activation('linear', name='doa_out')(doa)

    # FC - SED
    sed = spec_rnn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    model = None
    if doa_objective is 'mse':
        model = Model(inputs=spec_start, outputs=[sed, doa])
        model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)
    elif doa_objective is 'masked_mse':
        doa_concat = Concatenate(axis=-1, name='doa_concat')([sed, doa])
        model = Model(inputs=spec_start, outputs=[sed, doa_concat])
        model = compile_model(model, weights)
    else:
        print('ERROR: Unknown doa_objective: {}'.format(doa_objective))
        exit()
    model.summary()
    return model


def compile_model(model, weights):
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', masked_mse], loss_weights=weights)
    return model


def masked_mse(y_gt, model_out):
    sed_out = model_out[:, :, :11]
    sed_out = keras.backend.repeat_elements(sed_out, 2, -1)
    doa_out = model_out[:, :, 11:]
    return keras.losses.mse(y_gt, doa_out*sed_out)


def load_seld_model(model_file, doa_objective):
    if doa_objective is 'mse':
        return load_model(model_file)
    elif doa_objective is 'masked_mse':
        return load_model(model_file, custom_objects={'masked_mse': masked_mse})
    else:
        print('ERROR: Unknown doa objective: {}'.format(doa_objective))
        exit()



