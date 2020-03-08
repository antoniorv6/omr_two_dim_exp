from model.AttentionWanaSit import CreateAttentionModelWSHW
from model.S2S import CreateS2SModel

import cv2
import numpy as np
import os
import tqdm
import sys
import json

import tqdm

from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


def loadData(path):
    X = []
    Y = []

    print('Loading data...')

    for file in os.listdir(path):
        if file.endswith(".json"):
            with open(path + file) as json_file:
                originalImage = cv2.imread(os.path.splitext(path+file)[0], False)
                data = json.load(json_file)
                for page in data['pages']:
                    if "regions" in page:
                        for region in page['regions']:
                            if region['type'] == 'staff' and "symbols" in region:
                                symbol_sequence = [( s["agnostic_symbol_type"] + "-" + s["position_in_straff"], s["bounding_box"]["fromX"]) for s in region["symbols"]]
                                sorted_symbols = sorted(symbol_sequence, key=lambda symbol: symbol[1])
                                sequence = [sym[0] for sym in sorted_symbols]
                                Y.append(sequence)
                                top, left, bottom, right = region["bounding_box"]["fromY"], \
                                                           region["bounding_box"]["fromX"], \
                                                           region["bounding_box"]["toY"], \
                                                           region["bounding_box"]["toX"]
                                selected_region = originalImage[top:bottom, left:right]
                                if selected_region is not None:
                                    X.append(selected_region)

    print('Data loaded!')
    return np.array(X), np.array(Y)

def saveCheckpoint(modelToSave, fold, SER):
    modelToSave.save("checkpoints/handwritten/S2SA/model.h5")


def TrainLoop(model_to_train, X_train, Y_train, X_test, Y_test, FOLD):

    generator = batch_generator(X_train, Y_train, BATCH_SIZE)
    generatorTest = batch_generator(X_test, Y_test, BATCH_SIZE)

    best_value_eval = 190

    for epoch in range(100):
        print()
        print('----> Epoch', epoch * EVAL_EPOCH_STRIDE)

        history = model_to_train.fit_generator(generator,
                                      steps_per_epoch=len(X_train) // BATCH_SIZE,
                                      verbose=2,
                                      epochs=EVAL_EPOCH_STRIDE)

        current_val_ed = 0
        
        for step in range(len(X_test)//BATCH_SIZE):
            TestInputs, T_Test = next(generatorTest)
            batch_prediction = model_to_train.predict(TestInputs, batch_size=BATCH_SIZE)
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

        current_val_ed = (100. * current_val_ed) / len(X_test)
        
        nolookValEdition = 0

        print("Evaluating models...")
        for step in tqdm.tqdm(range(len(X_test)//BATCH_SIZE)):
            TestInputs, T_Test = next(generatorTest)
            for i, sequence in enumerate(TestInputs[0]):
                prediction = test_prediction(sequence, model_to_train, w2i, i2w)
                raw_gt = [i2w[char] for char in np.argmax(T_Test[i], axis=1)]

                gt = []
                for char in raw_gt:
                    gt += [char]
                    if char == '</s>':
                        break
            
                nolookValEdition += edit_distance(gt, prediction) / len(gt)

        print("Finished evaluation, displaying results") 
        valNoLookEdition = (100. * nolookValEdition) / len(X_test)
        
        print()
        print('Epoch ' + str(((epoch + 1) * EVAL_EPOCH_STRIDE) - 1) + ' - SER avg test with input: ' + str(current_val_ed))
        print('Epoch ' + str(((epoch + 1) * EVAL_EPOCH_STRIDE) - 1) + ' - SER avg test without input: ' + str(valNoLookEdition))
        print()

        if valNoLookEdition < best_value_eval:
            print("Saving best result")
            saveCheckpoint(model_to_train, FOLD, valNoLookEdition)
            best_value_eval = valNoLookEdition
    
    return best_value_eval

def save_vocabulary(fold):
    np.save("vocabulary/handwritten/S2SA/w2i.npy", w2i)
    np.save("vocabulary/handwritten/S2SA/i2w.npy", i2w)

if __name__ == "__main__":

    for i in range(1):
        fold = i+1
        RawX, RawY = loadData("dataHandwritten/B-59.850/")
        split = len(RawX)*0.2
        splitIndex = int(split/2)
        TestX = RawX[:splitIndex]
        ValidX = RawX[splitIndex:splitIndex*2]
        TrainX = RawX[splitIndex*2:]

        TestY = RawY[:splitIndex]
        ValidY = RawY[splitIndex:splitIndex*2]
        TrainY = RawY[splitIndex*2:]


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
        print("Saving vocabulary...")

        save_vocabulary(fold)

        model = CreateAttentionModelWSHW(FEATURESPERFRAME, ALPHABETLENGTH)

        bestvalue = TrainLoop(model, TrainX, Y_Train, TestX, Y_Test, fold)

        modelToTest = load_model("checkpoints/handwritten/S2SA/model.h5")

        print("TESTING MODEL ...")

        generatorValidation = batch_generator(ValidX, Y_Validate, BATCH_SIZE)
        #X_Validation, Y_Validation, T_Validation = batch_confection(ValidX, ValidY)
        validationEdition = 0
        for step in tqdm.tqdm(range(len(ValidX)//BATCH_SIZE)):
            ValidInputs, T_Validation = next(generatorValidation)
            for i, sequence in enumerate(ValidInputs[0]):
                prediction = test_prediction(sequence, modelToTest, w2i, i2w)
                raw_gt = [i2w[char] for char in np.argmax(T_Validation[i], axis=1)]

                gt = []
                for char in raw_gt:
                    gt += [char]
                    if char == '</s>':
                        break
            
                validationEdition += edit_distance(gt, prediction) / len(gt)

        print("Finished evaluation, displaying results") 
        displayResult = (100. * validationEdition) / len(ValidX)
        
        print("TRAINING BEST SER - " + str(bestvalue))
        print("SER WITH TEST DATA - " + str(displayResult))





