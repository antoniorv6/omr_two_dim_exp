from model.AttentionWanaSit import CreateAttentionModelWS
from model.S2S import CreateS2SModel
from model.S2S import CreateS2SModelHW
from model.AttentionWanaSit import CreateAttentionModelWSHW

import cv2
import numpy as np
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###CONSTANTS####
FOLDERPATH = 'dataHandwritten/'
FEATURESPERFRAME = 128
ALPHABETLENGTH = 0
w2i = {}
i2w = {}

BATCH_SIZE = 16
EVAL_EPOCH_STRIDE = 2
################


def edit_distance(a, b):
    n, m = len(a), len(b)

    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def batch_confection(batchX, batchY):
    max_image_len = max([image.shape[1] for image in batchX])

    encoder_input = np.zeros((len(batchX), FEATURESPERFRAME, max_image_len), dtype=np.float)
    for i, image in enumerate(batchX):
        encoder_input[i][:, :image.shape[1]] = image

    encoder_input = np.expand_dims(encoder_input, axis=-1)
    encoder_input = (255. - encoder_input) / 255.

    max_batch_output_len = max([len(sequence) for sequence in batchY])

    decoder_input = np.zeros((len(batchY), max_batch_output_len, ALPHABETLENGTH), dtype=np.float)
    decoder_output = np.zeros((len(batchY), max_batch_output_len, ALPHABETLENGTH), dtype=np.float)

    for i, sequence in enumerate(batchY):
        for j, char in enumerate(sequence):
            decoder_input[i][j][w2i[char]] = 1.
            if j > 1:
                decoder_output[i][j - 1][w2i[char]] = 1.

    return encoder_input, decoder_input, decoder_output


def batch_generator(X, Y, batch_size):
    index = 0
    while True:
        BatchX = X[index:index + batch_size]
        BatchY = Y[index:index + batch_size]

        encoder_input, decoder_input, decoder_output = batch_confection(BatchX, BatchY)

        yield [encoder_input, decoder_input], decoder_output

        index = (index + batch_size) % len(X)


def prepareVocabularyandOutput(rawY):
    global ALPHABETLENGTH
    global w2i, i2w

    output_sos = '<s>'
    output_eos = '</s>'

    Y = [[output_sos] + sequence + [output_eos] for sequence in rawY]

    # Setting up the vocabulary with positions and symbols
    vocabulary = set()

    for sequence in Y:
        vocabulary.update(sequence)

    ALPHABETLENGTH = len(vocabulary)

    # print('We have a total of ' + str(len(vocabulary)) + ' symbols')

    w2i = dict([(char, i) for i, char in enumerate(vocabulary)])
    i2w = dict([(i, char) for i, char in enumerate(vocabulary)])

    return Y


def resize_image(image):
    resize_width = int(float(FEATURESPERFRAME * image.shape[1]) / image.shape[0])
    return cv2.resize(image, (resize_width, FEATURESPERFRAME))


def loadData():
    X = []
    Y = []

    print('Loading data...')

    for folder in os.listdir(FOLDERPATH):
        if not os.path.isfile(folder):
            for file in os.listdir(FOLDERPATH + folder):
                if file.endswith(".json"):
                    with open(FOLDERPATH + folder + "/" + file) as json_file:
                        originalImage = cv2.imread(os.path.splitext(FOLDERPATH + folder + "/" + file)[0], False)
                        data = json.load(json_file)
                        for page in data['pages']:
                            if "regions" in page:
                                for region in page['regions']:
                                    if region['type'] == 'staff' and "symbols" in region:
                                        symbol_sequence = [s["agnostic_symbol_type"] + "-" + s["position_in_straff"] for s in region["symbols"]]
                                        Y.append(symbol_sequence)
                                        top, left, bottom, right = region["bounding_box"]["fromY"], \
                                                                   region["bounding_box"]["fromX"], \
                                                                   region["bounding_box"]["toY"], \
                                                                   region["bounding_box"]["toX"]

                                        selected_region = originalImage[top:bottom, left:right]
                                        if selected_region is not None:
                                            X.append(selected_region)

    print('Data loaded!')
    return np.array(X), np.array(Y)


def TrainLoop(model_to_train, X, Y, val_split):

    split_index = int(len(X) * val_split)
    X_train = X[split_index:]
    Y_train = Y[split_index:]

    X_validation = X[:split_index]
    Y_validation = Y[:split_index]

    generator = batch_generator(X_train, Y_train, BATCH_SIZE)

    X_Val, Y_Val, T_validation = batch_confection(X_validation, Y_validation)

    for epoch in range(30):
        print()
        print('----> Epoch', epoch * EVAL_EPOCH_STRIDE)

        history = model_to_train.fit_generator(generator,
                                      steps_per_epoch=len(X_train) // BATCH_SIZE,
                                      verbose=2,
                                      epochs=EVAL_EPOCH_STRIDE,
                                      validation_data=[[X_Val, Y_Val], T_validation])

        current_val_ed = 0
        batch_prediction = model_to_train.predict([X_Val, Y_Val], batch_size=BATCH_SIZE)

        for i, prediction in enumerate(batch_prediction):
            raw_sequence = [i2w[char] for char in np.argmax(prediction, axis=1)]
            raw_gt_sequence = [i2w[char] for char in np.argmax(T_validation[i], axis=1)]

            sequence = []
            gt = []

            for char in raw_sequence:
                sequence += [char]
                if char == '</s>':
                    break
            for char in raw_gt_sequence:
                gt += [char]
                if char == '</s>':
                    break

            current_val_ed += edit_distance(gt, sequence) / len(gt)

        current_val_ed = (100. * current_val_ed) / len(X_validation)
        print()
        print()
        print('Epoch ' + str(((epoch + 1) * EVAL_EPOCH_STRIDE) - 1) + ' - SER avg validation: ' + str(current_val_ed))
        print()
        print()


if __name__ == "__main__":

    RawX, RawY = loadData()

    for index, rimg in enumerate(RawX):
        RawX[index] = resize_image(rimg)

    print(RawX.shape)
    print(RawY.shape)

    RawY = prepareVocabularyandOutput(RawY)

    print(RawY[0])
    print("Vocabulary size: " + str(ALPHABETLENGTH))

    model = CreateS2SModelHW(FEATURESPERFRAME, ALPHABETLENGTH)

    TrainLoop(model, RawX, RawY, 0.1)



