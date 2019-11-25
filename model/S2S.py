from keras.layers import Input, Dense, Reshape, Permute, concatenate, Embedding, LSTM, dot, Activation, Concatenate, TimeDistributed
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras import backend as K

import numpy as np

#=== SIMPLE SEQ 2 SEQ MODEL ===#
def CreateS2SModel(FEATURESPERFRAME, ALPHABETLENGTH):
    filters = [32, 64, 128, 128]
    w_poolings = [2, 2, 2, 2]
    h_poolings = [2, 2, 2, 2]
    rnn_neurons = 512

    if K.image_data_format() == 'channels_last':
        input_data = Input(name='input', shape=(FEATURESPERFRAME, None, 1))
    else:
        input_data = Input(name='input', shape=(1, FEATURESPERFRAME, None))

    CNN = Conv2D(filters[0], (3, 3), padding='same', activation='relu')(input_data)
    CNN = BatchNormalization()(CNN)
    CNN = MaxPooling2D(pool_size=(h_poolings[0], w_poolings[0]))(CNN)

    CNN = Conv2D(filters[1], (3, 3), padding='same', activation='relu')(CNN)
    CNN = BatchNormalization()(CNN)
    CNN = MaxPooling2D(pool_size=(h_poolings[1], w_poolings[1]))(CNN)

    CNN = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(CNN)
    CNN = BatchNormalization()(CNN)
    CNN = MaxPooling2D(pool_size=(h_poolings[2], w_poolings[2]))(CNN)

    CNN = Conv2D(filters[3], (3, 3), padding='same', activation='relu')(CNN)
    CNN = BatchNormalization()(CNN)
    CNN = MaxPooling2D(pool_size=(h_poolings[3], w_poolings[3]))(CNN)

    permute = Permute((2, 1, 3))(CNN)
    w_factor = np.prod(np.array(w_poolings))
    h_factor = np.prod(np.array(h_poolings))

    conv_to_rnn_dims = (-1, (FEATURESPERFRAME // (h_factor)) * filters[3])
    encoder_input = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(permute)

    encoder_output, stateh, statec = LSTM(rnn_neurons, return_state=True)(encoder_input)
    states = [stateh, statec]

    decoder_inputs = Input(shape=(None, ALPHABETLENGTH))

    decoder_output, _, _ = LSTM(rnn_neurons, return_sequences=True, return_state=True)(decoder_inputs, initial_state=states)
    output = Dense(ALPHABETLENGTH, activation="softmax")(decoder_output)

    model = Model([input_data, decoder_inputs], output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary()

    return model


def CreateS2SModelHW(FEATURESPERFRAME, ALPHABETLENGTH):
    filters = [16, 32, 64, 128]
    w_poolings = [2, 2, 2, 2]
    h_poolings = [2, 2, 2, 2]
    rnn_neurons = 256

    if K.image_data_format() == 'channels_last':
        input_data = Input(name='input', shape=(FEATURESPERFRAME, None, 1))
    else:
        input_data = Input(name='input', shape=(1, FEATURESPERFRAME, None))

    CNN = Conv2D(filters[0], (3, 3), padding='same', activation='relu')(input_data)
    CNN = BatchNormalization()(CNN)
    CNN = MaxPooling2D(pool_size=(h_poolings[0], w_poolings[0]))(CNN)

    CNN = Conv2D(filters[1], (3, 3), padding='same', activation='relu')(CNN)
    CNN = BatchNormalization()(CNN)
    CNN = MaxPooling2D(pool_size=(h_poolings[1], w_poolings[1]))(CNN)

    CNN = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(CNN)
    CNN = BatchNormalization()(CNN)
    CNN = MaxPooling2D(pool_size=(h_poolings[2], w_poolings[2]))(CNN)

    CNN = Conv2D(filters[3], (3, 3), padding='same', activation='relu')(CNN)
    CNN = BatchNormalization()(CNN)
    CNN = MaxPooling2D(pool_size=(h_poolings[3], w_poolings[3]))(CNN)

    permute = Permute((2, 1, 3))(CNN)
    w_factor = np.prod(np.array(w_poolings))
    h_factor = np.prod(np.array(h_poolings))

    conv_to_rnn_dims = (-1, (FEATURESPERFRAME // (h_factor)) * filters[3])
    encoder_input = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(permute)

    encoder_output = LSTM(rnn_neurons, return_sequences=True, return_state=False)(encoder_input)
    encoder_output, stateh, statec = LSTM(rnn_neurons, return_state=True)(encoder_output)
    states = [stateh, statec]

    decoder_inputs = Input(shape=(None, ALPHABETLENGTH))

    decoder_output = LSTM(rnn_neurons, return_sequences=True)(decoder_inputs,
                                                                 initial_state=states)
    decoder_output, _, _ = LSTM(rnn_neurons, return_sequences=True, return_state=True)(decoder_output)

    output = Dense(ALPHABETLENGTH, activation="softmax")(decoder_output)

    model = Model([input_data, decoder_inputs], output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary()

    return model
