from keras.layers import Input, Dense, Reshape, Permute, concatenate, Embedding, LSTM, dot, Activation, Concatenate, TimeDistributed
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras import backend as K
import numpy as np

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

def CreateAttentionModelWS(FEATURESPERFRAME, ALPHABETLENGTH):

    init_session()

    filters = [32, 64, 128, 128]
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

    encoder_output, stateh, statec = LSTM(rnn_neurons, return_state=True, return_sequences=True)(encoder_input)
    states = [stateh, statec]

    decoder_inputs = Input(shape=(None, ALPHABETLENGTH))

    decoder_output, _, _ = LSTM(rnn_neurons, return_sequences=True, return_state=True)(decoder_inputs,
                                                                                       initial_state=states)
    # === ATTENTION MODEL ===#

    attention = dot([decoder_output, encoder_output], axes=[2,2])
    attention_mask = Activation('softmax')(attention)

    context = dot([attention_mask, encoder_output], axes=[2,1])
    decoder_combined_context = Concatenate()([context, decoder_output])


    output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context)

    # === END OF ATTENTION MODEL ===#

    output = TimeDistributed(Dense(ALPHABETLENGTH, activation="softmax"))(output)

    model = Model([input_data, decoder_inputs], output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary()

    return model

def CreateDualAttentionModel(FEATURESPERFRAME, ALPHABETSYMBOL, ALPHABETPOSITION):
    init_session()
    filters = [32, 64, 128, 128]
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

    encoder_output, stateh, statec = LSTM(rnn_neurons, return_state=True, return_sequences=True)(encoder_input)
    states = [stateh, statec]

    decoder_input_glyph = Input(shape=(None, ALPHABETSYMBOL))
    decoder_input_position = Input(shape=(None, ALPHABETPOSITION))

    decoder_output_symbol, _, _ = LSTM(rnn_neurons, return_sequences=True, return_state=True)(decoder_input_glyph,
                                                                                       initial_state=states)
    
    decoder_output_position, _, _ = LSTM(rnn_neurons, return_sequences=True, return_state=True)(decoder_input_position,
                                                                                       initial_state=states)

    ##Attention for glyph
    attention_glyph = dot([decoder_output_symbol, encoder_output], axes=[2,2])
    attention_mask_glyph = Activation('softmax')(attention_glyph)

    context_glyph = dot([attention_mask_glyph, encoder_output], axes=[2,1])
    decoder_combined_context_glyph = Concatenate()([context_glyph, decoder_output_symbol])

    ##Attention for position
    attention_position = dot([decoder_output_position, encoder_output], axes=[2,2])
    attention_mask_position = Activation('softmax')(attention_position)

    context_glyph = dot([attention_mask_position, encoder_output], axes=[2,1])
    decoder_combined_context_position = Concatenate()([context_glyph, decoder_output_position])

    tanhGlyph = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context_glyph)
    tanhPosition = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context_position)

    outputSymbol = TimeDistributed(Dense(ALPHABETSYMBOL, activation="softmax"))(tanhGlyph)
    outputPosition = TimeDistributed(Dense(ALPHABETSYMBOL, activation="softmax"))(tanhPosition)

    model = Model([input_data, decoder_input_glyph, decoder_input_position], [outputSymbol, outputPosition])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary()

    return model


def CreateAttentionModelWSHW(FEATURESPERFRAME, ALPHABETLENGTH):

    filters = [16, 64, 128, 128]
    w_poolings = [2, 2, 2, 2]
    h_poolings = [2, 2, 2, 2]
    rnn_neurons = 128

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

    encoder_output, stateh, statec = LSTM(rnn_neurons, return_state=True, return_sequences=True)(encoder_input)
    states = [stateh, statec]

    decoder_inputs = Input(shape=(None, ALPHABETLENGTH))

    decoder_output, _, _ = LSTM(rnn_neurons, return_sequences=True, return_state=True)(decoder_inputs,
                                                                                       initial_state=states)
    # === ATTENTION MODEL ===#

    attention = dot([decoder_output, encoder_output], axes=[2,2])
    attention_mask = Activation('softmax')(attention)

    context = dot([attention_mask, encoder_output], axes=[2,1])
    decoder_combined_context = Concatenate()([context, decoder_output])


    output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context)

    # === END OF ATTENTION MODEL ===#

    output = TimeDistributed(Dense(ALPHABETLENGTH, activation="softmax"))(output)

    model = Model([input_data, decoder_inputs], output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary()

    return model


def init_session():
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth= True
    sess = tf.Session(config=conf)
    set_session(sess)