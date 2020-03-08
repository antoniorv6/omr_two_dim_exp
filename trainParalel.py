from model.AttentionWanaSit import CreateAttentionModelWS, CreateDualAttentionModel
from model.S2S import CreateS2SModel

import cv2
import numpy as np
import os
import tqdm
import sys

import tqdm

from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

###CONSTANTS####
FOLDERPATH = 'CameraPrimus/'
FEATURESPERFRAME = 128
ALPHABETLENGTHGLYPH = 0
ALPHABETLENGTHPOSITION = 0
w2iglyph = {}
i2wglyph = {}
w2iposition = {}
i2wposition = {}

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
    decoded = np.zeros((1,500,ALPHABETLENGTHPOSITION), dtype=np.float)
    decoded_input = np.asarray(decoded)
    prediction = model.predict([[sequence], decoded_input])
    predicted_sequence = [i2w[char] for char in np.argmax(prediction[0], axis=1)]
    predicted = []
    
    for char in predicted_sequence:
        predicted += [char]
        if char == '</s>':
            break

    return predicted


def test_predictiondual(sequence, model):
    decoded_symbols = np.zeros((1,500,ALPHABETLENGTHGLYPH), dtype=np.float)
    decoded_position = np.zeros((1,500,ALPHABETLENGTHPOSITION), dtype=np.float)

    decoded_symbols = np.asarray(decoded_symbols)
    decoded_position = np.asarray(decoded_position)


    prediction_glyph, prediction_pos = model.predict([[sequence], decoded_symbols, decoded_position])
    
    predicted_sequence_glyph = [i2wglyph[char] for char in np.argmax(prediction_glyph[0], axis=1)]
    predicted_sequence_pos = [i2wposition[char] for char in np.argmax(prediction_pos[0], axis=1)]

    predicted = []
    
    for i, char in enumerate(predicted_sequence_glyph):
        character = [char + "-" + predicted_sequence_pos[i]]
        predicted += character
        if character[0] == "</s>-</s>":
            break

    return predicted
    

def batch_confection(batchX, batchYglyph, batchyPosition):
    max_image_len = max([image.shape[1] for image in batchX])

    encoder_input = np.zeros((len(batchX), FEATURESPERFRAME, max_image_len), dtype=np.float)
    for i, image in enumerate(batchX):
        encoder_input[i][:, :image.shape[1]] = image

    encoder_input = np.expand_dims(encoder_input, axis=-1)
    encoder_input = (255. - encoder_input) / 255.

    max_batch_output_len_glyph = max([len(sequence) for sequence in batchYglyph])
    max_batch_output_len_pos = max([len(sequence) for sequence in batchyPosition])

    decoder_input_glyph = np.zeros((len(batchYglyph), max_batch_output_len_glyph, ALPHABETLENGTHGLYPH), dtype=np.float)
    decoder_input_pos = np.zeros((len(batchyPosition), max_batch_output_len_glyph, ALPHABETLENGTHPOSITION), dtype=np.float)

    decoder_output_glyph = np.zeros((len(batchYglyph), max_batch_output_len_pos, ALPHABETLENGTHGLYPH), dtype=np.float)
    decoder_output_pos = np.zeros((len(batchYglyph), max_batch_output_len_pos, ALPHABETLENGTHPOSITION), dtype=np.float)

    for i, sequence in enumerate(batchYglyph):
        for j, char in enumerate(sequence):
            if j > 0:
                decoder_output_glyph[i][j - 1][w2iglyph[char]] = 1.

    for i, sequence in enumerate(batchyPosition):
        for j, posChar in enumerate(sequence):
            if j > 0:
                decoder_output_pos[i][j - 1][w2iposition[posChar]] = 1.

    return encoder_input, decoder_input_glyph, decoder_input_pos, decoder_output_glyph, decoder_output_pos


def batch_generator(X, YGlyph, YPos, batch_size):
    index = 0
    while True:
        BatchX = X[index:index + batch_size]

        BatchYGlyph = YGlyph[index:index + batch_size]
        BatchYPosition = YPos[index:index + batch_size]

        encoder_input, decoder_input_glyph, decoder_input_pos, decoder_output_glyph, decoder_output_pos = batch_confection(BatchX, BatchYGlyph, BatchYPosition)

        yield [encoder_input, decoder_input_glyph, decoder_input_pos], [decoder_output_glyph, decoder_output_pos]

        index = (index + batch_size) % len(X)


def prepareVocabularyandOutput(trainYglyph, trainYpos, testYglyph, testYpos, valYglyph, valYpos):
    global ALPHABETLENGTHGLYPH, ALPHABETLENGTHPOSITION
    global w2iglyph, w2iposition, i2wglyph, i2wposition

    output_sos = '<s>'
    output_eos = '</s>'

    Y_trainGlyph = [[output_sos] + sequence + [output_eos] for sequence in trainYglyph]
    Y_testGlyph = [[output_sos] + sequence + [output_eos] for sequence in testYglyph]
    Y_valGlyph = [[output_sos] + sequence + [output_eos] for sequence in valYglyph]

    Y_trainPos = [[output_sos] + sequence + [output_eos] for sequence in trainYpos]
    Y_testPos = [[output_sos] + sequence + [output_eos] for sequence in testYpos]
    Y_valPos = [[output_sos] + sequence + [output_eos] for sequence in valYpos]

    # Setting up the vocabulary with positions and symbols
    vocabularyGlyph = set()
    vocabularyPos = set()

    for sequence in Y_trainGlyph:
        vocabularyGlyph.update(sequence)
    for sequence in Y_testGlyph:
        vocabularyGlyph.update(sequence)
    for sequence in Y_valGlyph:
        vocabularyGlyph.update(sequence)

    for sequence in Y_trainPos:
        vocabularyPos.update(sequence)
    for sequence in Y_testPos:
        vocabularyPos.update(sequence)
    for sequence in Y_valPos:
        vocabularyPos.update(sequence)

    ALPHABETLENGTHGLYPH = len(vocabularyGlyph) + 1
    ALPHABETLENGTHPOSITION = len(vocabularyPos) + 1

    # print('We have a total of ' + str(len(vocabulary)) + ' symbols')

    w2iglyph = dict([(char, i+1) for i, char in enumerate(vocabularyGlyph)])
    i2wglyph = dict([(i+1, char) for i, char in enumerate(vocabularyGlyph)])
    w2iposition = dict([(char, i+1) for i, char in enumerate(vocabularyPos)])
    i2wposition = dict([(i+1, char) for i, char in enumerate(vocabularyPos)])

    w2iglyph['PAD'] = 0
    i2wglyph[0] = 'PAD'
    w2iposition['PAD'] = 0
    i2wposition[0] = 'PAD'

    return Y_trainGlyph, Y_testGlyph, Y_valGlyph, Y_trainPos, Y_testPos, Y_valPos 


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

def parseSequence(sequence):
    parsed = []
    for char in sequence:
        parsed += char.split("-")
    return parsed

def parseCharacter(char):
    parsed = char.split("-")
    return parsed

def loadDataCP(filepath, samples):
    X = []
    YParseGlyph = []
    YParsePosition = []
    YGlyph = []
    YPos = []

    currentsamples = 0

    with open(filepath, "r") as datafile:
        line = datafile.readline()
        while line:
            files = line.split()

            image = cv2.imread(files[0], False)
            sequenceFile = open(files[1], "r")

            X.append(image)
            for char in sequenceFile.readline().split():
                chars = parseCharacter(char)
                YParseGlyph += [chars[0]]
                YParsePosition += [chars[1]]
            
            YGlyph.append(YParseGlyph)
            YPos.append(YParsePosition)
            YParseGlyph = []
            YParsePosition = []

            sequenceFile.close()
            line = datafile.readline()

            currentsamples += 1

            if currentsamples == samples:
                datafile.close()
                break

    return np.array(X), np.array(YGlyph), np.array(YPos)

def saveCheckpoint(modelToSave, fold, SER):
    modelToSave.save("checkpoints/FOLD" + str(fold) + "/model.h5")


def TrainLoop(model_to_train, X_train, Y_trainGlyph, Y_trainPosition, X_test, Y_testGlyph, Y_testPosition, FOLD):

    generator = batch_generator(X_train, Y_trainGlyph, Y_trainPosition, BATCH_SIZE)
    generatorTest = batch_generator(X_test, Y_testGlyph, Y_testPosition, BATCH_SIZE)

    best_value_eval = 190

    for epoch in range(30):
        print()
        print('----> Epoch', epoch * EVAL_EPOCH_STRIDE)

        _ = model_to_train.fit_generator(generator,
                                      steps_per_epoch=len(X_train) // BATCH_SIZE,
                                      verbose=2, 
                                      epochs=EVAL_EPOCH_STRIDE)

        current_val_ed = 0
        
        for step in range(len(X_test)//BATCH_SIZE):
            TestInputs, T_Test = next(generatorTest)
            prediction_symbols, prediction_position = model_to_train.predict(TestInputs, batch_size=BATCH_SIZE)
            TestGlyph = T_Test[0]
            TestPosition = T_Test[1]
            
            for i, prediction in enumerate(prediction_symbols):
                raw_symbols = [i2wglyph[char] for char in np.argmax(prediction, axis=1)]
                raw_position = [i2wposition[char] for char in np.argmax(prediction_position[i], axis=1)]
                raw_gt_symbols = [i2wglyph[char] for char in np.argmax(TestGlyph[i], axis=1)]
                raw_gt_position = [i2wposition[char] for char in np.argmax(TestPosition[i], axis=1)]

                sequence = []
                gt = []

                for i, char in enumerate(raw_symbols):
                    character = [char + "-" + raw_position[i]]
                    sequence += character
                    if character[0] == "</s>-</s>":
                        break
                for i, char in enumerate(raw_gt_symbols):
                    character = [char + "-" + raw_gt_position[i]]
                    gt += character
                    if character[0] == "</s>-</s>":
                        break

                current_val_ed += edit_distance(gt, sequence) / len(gt)

        current_val_ed = (100. * current_val_ed) / len(X_test)
        
        nolookValEdition = 0

        print("Evaluating models...")
        for step in range(len(X_test)//BATCH_SIZE):
            TestInputs, T_Test = next(generatorTest)
            TestGlyph = T_Test[0]
            TestPosition = T_Test[1]

            for i, sequence in enumerate(TestInputs[0]):
                prediction = test_predictiondual(sequence, model_to_train)
                raw_gt_symbols = [i2wglyph[char] for char in np.argmax(TestGlyph[i], axis=1)]
                raw_gt_position = [i2wposition[char] for char in np.argmax(TestPosition[i], axis=1)]

                gt = []
                for i, char in enumerate(raw_gt_symbols):
                    character = [char + "-" + raw_gt_position[i]]
                    gt += character
                    if character[0] == "</s>-</s>":
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
    np.save("vocabulary/paralel/w2iglyph.npy", w2iglyph)
    np.save("vocabulary/paralel/i2wglyph.npy", i2wglyph)
    np.save("vocabulary/paralel/w2iposition.npy", w2iposition)
    np.save("vocabulary/paralel/i2wposition.npy", i2wposition)

if __name__ == "__main__":

    for i in range(1):
        fold = i+1
        print("WORKING ON FOLD " + str(fold))
        print("Loading training data...")
        TrainX, TrainYGlyph, TrainYPos = loadDataCP("./CameraPrimusFolds/Fold"+ str(fold) +"/train", 30000)
        print("Loading testing data...")
        TestX, TestYGlyph, TestYPos = loadDataCP("./CameraPrimusFolds/Fold"+ str(fold) +"/test", 10000)
        print("Loading validation data...")
        ValidX, ValidYGlyph, ValidYPos = loadDataCP("./CameraPrimusFolds/Fold"+ str(fold) +"/validation", 10000)

        for index, rimg in enumerate(TrainX):
            TrainX[index] = resize_image(rimg)
    
        for index, rimg in enumerate(TestX):
            TestX[index] = resize_image(rimg)
    
        for index, rimg in enumerate(ValidX):
            ValidX[index] = resize_image(rimg)

        print("//// - TRAINING DATA - ////")
        print(TrainX.shape)
        print(TrainYGlyph.shape)
        print(TrainYPos.shape)
        print("///////////////////////////")
        print()
        print("//// - TESTING DATA - ////")
        print(TestX.shape)
        print(TestYGlyph.shape)
        print(TestYPos.shape)
        print("///////////////////////////")
        print()
        print("//// - VALIDATION DATA - ////")
        print(ValidX.shape)
        print(ValidYGlyph.shape)
        print(ValidYPos.shape)
        print("///////////////////////////")

        print(TrainYGlyph[0])
        print(TrainYPos[0])

        Y_TrainGlyph, Y_TestGlyph, Y_ValidateGlyph, Y_TrainPos, Y_TestPos, Y_ValidatePos = prepareVocabularyandOutput(TrainYGlyph, TrainYPos, TestYGlyph, TestYPos, ValidYGlyph, ValidYPos)


        print("Vocabulary size glyphs: " + str(ALPHABETLENGTHGLYPH))
        print("Vocabulary size position: " + str(ALPHABETLENGTHPOSITION))

        print("Saving vocabulary...")

        save_vocabulary(fold)

        model = CreateDualAttentionModel(FEATURESPERFRAME, ALPHABETLENGTHGLYPH, ALPHABETLENGTHPOSITION)

        bestvalue = TrainLoop(model, TrainX, Y_TrainGlyph, Y_TrainPos, TestX, Y_TestGlyph, Y_TestPos, fold)

        modelToTest = load_model("checkpoints/FOLD" + str(fold) + "/model.h5")

        print("TESTING MODEL ...")

        generatorValidation = batch_generator(ValidX, Y_ValidateGlyph, Y_ValidatePos, BATCH_SIZE)
        #X_Validation, Y_Validation, T_Validation = batch_confection(ValidX, ValidY)
        validationEdition = 0
        for step in tqdm.tqdm(range(len(ValidX)//BATCH_SIZE)):
            ValidInputs, T_Validation = next(generatorValidation)
            ValidationGlyph = T_Validation[0]
            ValidationPosition = T_Validation[1]
            batch_prediction = modelToTest.predict(ValidInputs, batch_size=BATCH_SIZE)
            for i, sequence in enumerate(ValidInputs[0]):
                prediction = test_predictiondual(sequence, modelToTest)
                raw_gt_symbols = [i2wglyph[char] for char in np.argmax(ValidationGlyph[i], axis=1)]
                raw_gt_position = [i2wposition[char] for char in np.argmax(ValidationPosition[i], axis=1)]

                gt = []
                for i, char in enumerate(raw_gt_symbols):
                    character = [char + "-" + raw_gt_position[i]]
                    gt += character
                    if character[0] == "</s>-</s>":
                        break
            
                validationEdition += edit_distance(gt, prediction) / len(gt)

        print("Finished evaluation, displaying results") 
        displayResult = (100. * validationEdition) / len(ValidX)
        
        file = open("checkpoints/FOLD" + str(fold) + "/resume.txt", "w+")
        file.write("MODEL RESULTS IN FOLD " + str(fold) + "\n")
        file.write("TRAINING BEST SER - " + str(bestvalue) + "\n")
        file.write("SER WITH TEST DATA - " + str(displayResult))
        file.close()





