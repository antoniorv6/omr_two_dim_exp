from model.AttentionWanaSit import CreateAttentionModelWS
from model.S2S import CreateS2SModel
from utils.utils import edit_distance, make_single_prediction, LoadCameraPrimus, prepareOutput1, resize_image, saveCheckpointKeras, parseSequence

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
BATCH_SIZE = 8
EVAL_EPOCH_STRIDE = 2
################    

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


def TrainLoop(model_to_train, X_train, Y_train, X_test, Y_test, FOLD, w2i, i2w, ALPHABETLENGTH):

    generator = batch_generator(X_train, Y_train, BATCH_SIZE)
    generatorTest = batch_generator(X_test, Y_test, BATCH_SIZE)

    best_value_eval = 190

    for epoch in range(30):
        print()
        print('----> Epoch', epoch * EVAL_EPOCH_STRIDE)

        history = model_to_train.fit_generator(generator,
                                      steps_per_epoch=len(X_train) // BATCH_SIZE,
                                      verbose=2,
                                      epochs=EVAL_EPOCH_STRIDE)

        current_val_ed = 0
        
        for step in range(len(X_test)//BATCH_SIZE):
            TestInputs, T_Test = next(generatorTest)
            # batch_prediction = model_to_train.predict(TestInputs, batch_size=BATCH_SIZE)
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
                
                sequence = parseSequence(sequence)
                gt = parseSequence(gt)

                current_val_ed += edit_distance(gt, sequence) / len(gt)

        current_val_ed = (100. * current_val_ed) / len(X_test)
        
        nolookValEdition = 0

        for step in tqdm.tqdm(range(len(X_test)//BATCH_SIZE)):
            TestInputs, T_Test = next(generatorTest)
            for i, sequence in enumerate(TestInputs[0]):
                prediction = make_single_prediction(sequence, model_to_train, i2w, ALPHABETLENGTH)
                raw_gt = [i2w[char] for char in np.argmax(T_Test[i], axis=1)]

                gt = []
                for char in raw_gt:
                    gt += [char]
                    if char == '</s>':
                        break
                
                prediction = parseSequence(prediction)
                gt = parseSequence(gt)
                
                nolookValEdition += edit_distance(gt, prediction) / len(gt)

        valNoLookEdition = (100. * nolookValEdition) / len(X_test)
        
        print()
        print('Epoch ' + str(((epoch + 1) * EVAL_EPOCH_STRIDE) - 1) + ' - SER avg test with input: ' + str(current_val_ed))
        print('Epoch ' + str(((epoch + 1) * EVAL_EPOCH_STRIDE) - 1) + ' - SER avg test without input: ' + str(valNoLookEdition))
        print()

        if valNoLookEdition < best_value_eval:
            print("Saving best result")
            saveCheckpointKeras(model_to_train, "checkpoints/Printed/S2SA/", 0, fold)
            best_value_eval = valNoLookEdition
    
    return best_value_eval

if __name__ == "__main__":

    for i in range(5):
        fold = i + 1
        print("WORKING ON FOLD " + str(fold))
        print("Loading training data...")
        TrainX, TrainY = LoadCameraPrimus("./CameraPrimusFolds/Fold"+ str(fold) +"/train", 30000)
        print("Loading testing data...")
        TestX, TestY = LoadCameraPrimus("./CameraPrimusFolds/Fold"+ str(fold) +"/test", 10000)
        print("Loading validation data...")
        ValidX, ValidY = LoadCameraPrimus("./CameraPrimusFolds/Fold"+ str(fold) +"/validation", 10000)

        for index, rimg in enumerate(TrainX):
            TrainX[index] = resize_image(rimg, FEATURESPERFRAME)
    
        for index, rimg in enumerate(TestX):
            TestX[index] = resize_image(rimg, FEATURESPERFRAME)
    
        for index, rimg in enumerate(ValidX):
            ValidX[index] = resize_image(rimg, FEATURESPERFRAME)

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

        w2i = {}
        i2w = {}

        Y_Train, Y_Test, Y_Validate, w2i, i2w, ALPHABETLENGTH = prepareOutput1(TrainY, TestY, ValidY, i2w, w2i, "Printed", fold)

        print("Vocabulary size: " + str(ALPHABETLENGTH))

        model = CreateAttentionModelWS(FEATURESPERFRAME, ALPHABETLENGTH)

        bestvalue = TrainLoop(model, TrainX, Y_Train, TestX, Y_Test, fold, w2i, i2w, ALPHABETLENGTH)
        
        modelToTest = load_model("checkpoints/Printed/S2SA/modelc0fold"+str(fold)+".h5")

        print("TESTING MODEL ...")

        generatorValidation = batch_generator(ValidX, Y_Validate, BATCH_SIZE)
        #X_Validation, Y_Validation, T_Validation = batch_confection(ValidX, ValidY)
        validationEdition = 0
        for step in tqdm.tqdm(range(len(ValidX)//BATCH_SIZE)):
            ValidInputs, T_Validation = next(generatorValidation)
            batch_prediction = modelToTest.predict(ValidInputs, batch_size=BATCH_SIZE)
            for i, sequence in enumerate(ValidInputs[0]):
                prediction = make_single_prediction(sequence, modelToTest, w2i, i2w)
                raw_gt = [i2w[char] for char in np.argmax(T_Validation[i], axis=1)]

                gt = []
                for char in raw_gt:
                    gt += [char]
                    if char == '</s>':
                        break
                
                prediction = parseSequence(prediction)
                gt = parseSequence(gt)

                validationEdition += edit_distance(gt, prediction) / len(gt)

        displayResult = (100. * validationEdition) / len(ValidX)
        
        print("FOLD " + str(fold) + " SER: " + str(displayResult))





