from model.AttentionWanaSit import CreateAttentionModelWS
from model.S2S import CreateS2SModel

import cv2
import numpy as np
import os
import tqdm
import sys

from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

###CONSTANTS####
FOLDERPATH = 'CameraPrimus/'
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

def test_prediction(sequence, model, w2itarget, i2wtarget):
    decoded = np.zeros((1,500,ALPHABETLENGTH), dtype=np.float)
    decoded_input = np.asarray(decoded)
    prediction = model.predict([[sequence], decoded_input])
    predicted_sequence = [i2w[char] for char in np.argmax(prediction[0], axis=1)]
    predicted = []
    
    for char in predicted_sequence:
        predicted += [char]
        if char == '</s>':
            break

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
            if j > 0:
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


def parseSequence(sequence):
    
    parsed = []
    for char in sequence:
        parsed += char.split("-")
    return parsed

def prepareVocabularyandOutput(trainY, testY, valY):
    global ALPHABETLENGTH
    global w2i, i2w

    output_sos = '<s>'
    output_eos = '</s>'

    Y_train = [[output_sos] + parseSequence(sequence) + [output_eos] for sequence in trainY]
    Y_test = [[output_sos] + parseSequence(sequence) + [output_eos] for sequence in testY]
    Y_val = [[output_sos] + parseSequence(sequence) + [output_eos] for sequence in valY]

    # Setting up the vocabulary with positions and symbols
    vocabulary = set()

    for sequence in Y_train:
        vocabulary.update(sequence)
    for sequence in Y_test:
        vocabulary.update(sequence)
    for sequence in Y_val:
        vocabulary.update(sequence)

    ALPHABETLENGTH = len(vocabulary) + 1

    # print('We have a total of ' + str(len(vocabulary)) + ' symbols')

    w2i = dict([(char, i+1) for i, char in enumerate(vocabulary)])
    i2w = dict([(i+1, char) for i, char in enumerate(vocabulary)])

    w2i['PAD'] = 0
    i2w[0] = 'PAD'

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

def saveCheckpoint(modelToSave, fold, SER):
    modelToSave.save("checkpoints/FOLD" + str(fold) + "/checkpoint" + str(SER))


def TrainLoop(model_to_train, X_train, Y_train, X_test, Y_test, FOLD):

    generator = batch_generator(X_train, Y_train, BATCH_SIZE)

    X_Test, Y_Test, T_Test = batch_confection(X_test, Y_test)

    best_value_eval = 190

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

        print("Evaluating models...")
        for i, sequence in enumerate(X_Test):
            prediction = test_prediction(sequence, model_to_train, w2i, i2w)
            #print("Prediction done")
            raw_gt = [i2w[char] for char in np.argmax(T_Test[i], axis=1)]

            gt = []
            for char in raw_gt:
                gt += [char]
                if char == '</s>':
                    break
            
            nolookValEdition += edit_distance(gt, prediction) / len(gt)

        print("Finished evaluation, displaying results") 
        valNoLookEdition = (100. * nolookValEdition) / len(X_Test)
        
        print()
        print('Epoch ' + str(((epoch + 1) * EVAL_EPOCH_STRIDE) - 1) + ' - SER avg test with input: ' + str(current_val_ed))
        print('Epoch ' + str(((epoch + 1) * EVAL_EPOCH_STRIDE) - 1) + ' - SER avg test without input: ' + str(valNoLookEdition))
        print()

        if valNoLookEdition < best_value_eval:
            print("Saving best result")
            saveCheckpoint(model_to_train, FOLD, valNoLookEdition)
            best_value_eval = valNoLookEdition
    
    return best_value_eval


if __name__ == "__main__":

    for i in range(4):
        fold = i+1
        print("WORKING ON FOLD " + str(fold))
        print("Loading training data...")
        TrainX, TrainY = loadDataCP("./CameraPrimusFolds/Fold"+ str(fold) +"/train", 20000)
        print("Loading testing data...")
        TestX, TestY = loadDataCP("./CameraPrimusFolds/Fold"+ str(fold) +"/test", 10000)
        print("Loading validation data...")
        ValidX, ValidY = loadDataCP("./CameraPrimusFolds/Fold"+ str(fold) +"/validation", 10000)

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
        print(TestY)
        print("///////////////////////////")
        print()
        print("//// - VALIDATION DATA - ////")
        print(ValidX.shape)
        print(ValidY.shape)
        print("///////////////////////////")

        Y_Train, Y_Test, Y_Validate = prepareVocabularyandOutput(TrainY, TestY, ValidY)

        print("Vocabulary size: " + str(ALPHABETLENGTH))

        model = CreateAttentionModelWS(FEATURESPERFRAME, ALPHABETLENGTH)

        bestvalue = TrainLoop(model, TrainX, Y_Train, TestX, Y_Test, fold)

        modelToTest = load_model("checkpoints/FOLD" + str(fold) + "/checkpoint/model.mk5")

        print("TESTING MODEL ...")

        X_Validation, Y_Validation, T_Validation = batch_confection(ValidX, ValidY)
        
        validationEdition = 0
        for i, sequence in enumerate(X_Validation):
            prediction = test_prediction(sequence, modelToTest, w2i, i2w)
            #print("Prediction done")
            raw_gt = [i2w[char] for char in np.argmax(T_Validation[i], axis=1)]

            gt = []
            for char in raw_gt:
                gt += [char]
                if char == '</s>':
                    break
            
            validationEdition += edit_distance(gt, prediction) / len(gt)

        print("Finished evaluation, displaying results") 
        displayResult = (100. * validationEdition) / len(X_Validation)
        
        print()
        print('SER in final validation: ' + str(displayResult))
        print()





