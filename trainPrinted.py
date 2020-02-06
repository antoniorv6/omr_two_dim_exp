from model.AttentionWanaSit import CreateAttentionModelWS
from model.S2S import CreateS2SModel

import cv2
import numpy as np
import os
import tqdm
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###CONSTANTS####
FOLDERPATH = 'CameraPrimus/'
FEATURESPERFRAME = 64
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

def test_prediction(sequence, model, w2itarget, i2wtarget):
    decoded = np.zeros((1,1,ALPHABETLENGTH))
    decoded[0,0, w2i['<s>']] = 1.0
    predicted = []
    
    for i in range(1, 500):
        decoded_input = np.asarray(decoded)

        prediction = model.predict([[sequence], decoded_input])
        
        decoded = np.append(decoded, np.zeros((1, 1, ALPHABETLENGTH)), axis=0)

        predictedWord = np.argmax(prediction[0][-1])

        if i2wtarget[predictedWord] == '</s>':
            break
        
        predicted.append(i2wtarget[predictedWord])

    return predicted

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


def prepareVocabularyandOutput(trainY, testY, valY):
    global ALPHABETLENGTH
    global w2i, i2w

    output_sos = '<s>'
    output_eos = '</s>'

    Y_train = [[output_sos] + sequence + [output_eos] for sequence in trainY]
    Y_test = [[output_sos] + sequence + [output_eos] for sequence in testY]
    Y_val = [[output_sos] + sequence + [output_eos] for sequence in valY]

    # Setting up the vocabulary with positions and symbols
    vocabulary = set()

    for sequence in Y_train:
        vocabulary.update(sequence)
    for sequence in Y_test:
        vocabulary.update(sequence)
    for sequence in Y_val:
        vocabulary.update(sequence)

    ALPHABETLENGTH = len(vocabulary)

    # print('We have a total of ' + str(len(vocabulary)) + ' symbols')

    w2i = dict([(char, i) for i, char in enumerate(vocabulary)])
    i2w = dict([(i, char) for i, char in enumerate(vocabulary)])

    return Y_train, Y_test, Y_val


def resize_image(image):
    resize_width = int(float(FEATURESPERFRAME * image.shape[1]) / image.shape[0])
    return cv2.resize(image, (resize_width, FEATURESPERFRAME))


def loadData():
    X = []
    Y = []

    samples = 20000

    print('Loading data...')
    for i in tqdm.tqdm(range(1, samples + 1)):
        image = cv2.imread(FOLDERPATH + "/" + str(i) + ".png", False)
        sequenceFile = open(FOLDERPATH + "/" + str(i) + ".txt", "r")
        sequence = sequenceFile.readline().split()
        sequenceFile.close()
        X.append(image)
        Y.append(sequence)
    print('Data loaded!')
    return np.array(X), np.array(Y)

def loadDataCP(filepath, samples):
    X = []
    Y = []

    currentsamples = 0

    with open(filepath, "r") as datafile:
        line = datafile.readline()
        while line:
            files = line.split()

            image = cv2.imread(files[0], False)
            sequenceFile = open(files[1], "r")

            X.append(image)
            Y.append(sequenceFile.readline().split())

            sequenceFile.close()
            line = datafile.readline()

            currentsamples += 1

            if currentsamples == samples:
                datafile.close()
                break

    return np.array(X), np.array(Y)


def TrainLoop(model_to_train, X_train, Y_train, X_test, Y_test):

    generator = batch_generator(X_train, Y_train, BATCH_SIZE)

    X_Test, Y_Test, T_Test = batch_confection(X_test, Y_test)

    for epoch in range(30):
        print()
        print('----> Epoch', epoch * EVAL_EPOCH_STRIDE)

        history = model_to_train.fit_generator(generator,
                                      steps_per_epoch=len(X_train) // BATCH_SIZE,
                                      verbose=2,
                                      epochs=EVAL_EPOCH_STRIDE)

        current_val_ed = 0
        batch_prediction = model_to_train.predict([X_Test, Y_Test], batch_size=BATCH_SIZE)

        for i, prediction in enumerate(batch_prediction):
            raw_sequence = [i2w[char] for char in np.argmax(prediction, axis=1)]
            raw_gt_sequence = [i2w[char] for char in np.argmax(T_Test[i], axis=1)]

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

        current_val_ed = (100. * current_val_ed) / len(X_Test)
        
        nolookValEdition = 0

        for i, sequence in enumerate(X_Test):
            prediction = test_prediction(sequence, model_to_train, w2i, i2w)
            raw_gt = [i2w[char] for char in np.argmax(T_Test[i], axis=1)]

            gt = []
            for char in raw_gt:
                gt += [char]
                if char == '</s>':
                    break
            
            nolookValEdition += edit_distance(gt, prediction) / len(gt)
        
        valNoLookEdition = (100. * nolookValEdition) / len(X_Test)
        
        print()
        print()
        print('Epoch ' + str(((epoch + 1) * EVAL_EPOCH_STRIDE) - 1) + ' - SER avg test with input: ' + str(current_val_ed))
        print('Epoch ' + str(((epoch + 1) * EVAL_EPOCH_STRIDE) - 1) + ' - SER avg test without input: ' + str(valNoLookEdition))
        print()
        print()


if __name__ == "__main__":

    print("Loading training data...")
    TrainX, TrainY = loadDataCP("./CameraPrimusFolds/Fold1/train", 30000)
    print("Loading testing data...")
    TestX, TestY = loadDataCP("./CameraPrimusFolds/Fold1/test", 5000)
    print("Loading validation data...")
    ValidX, ValidY = loadDataCP("./CameraPrimusFolds/Fold1/validation", 5000)

    for index, rimg in enumerate(TrainX):
        TrainX[index] = resize_image(rimg)
    for index, rimg in enumerate(TestX):
        TestX[index] = resize_image(rimg)
    for index, rimg in enumerate(ValidX):
        ValidX[index] = resize_image(rimg)

    print("//// - TRAINING DATA - ////")
    print(TrainX.shape)
    print(TrainY.shape)
    print("///////////////////////////")
    print()
    print("//// - TESTING DATA - ////")
    print(TestX.shape)
    print(TestY.shape)
    print("///////////////////////////")
    print()
    print("//// - VALIDATION DATA - ////")
    print(ValidX.shape)
    print(ValidY.shape)
    print("///////////////////////////")

    Y_Train, Y_Test, Y_Validate = prepareVocabularyandOutput(TrainY, TestY, ValidY)

    print("Vocabulary size: " + str(ALPHABETLENGTH))

    model = CreateS2SModel(FEATURESPERFRAME, ALPHABETLENGTH)

    TrainLoop(model, TrainX, TrainY, TestX, TestY)




