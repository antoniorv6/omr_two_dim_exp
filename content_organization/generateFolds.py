
import tqdm
import numpy as np
import os

# ==== CONSTANTS ====
NUMFOLDS = 5
TESTPERC = 0.15
VALIDATIONPERC = 0.15
# ===================


def vector_to_file(filename, data):
    with open(filename, "w") as file:
        for writedata in data:
            file.write(writedata)
    file.close()


def loadMetaData():
    data = []

    with open('../dataset.lst', "r") as dataFile:
        line = dataFile.readline()
        while line:
           data.append(line)
           line = dataFile.readline()

        dataFile.close()

    return np.array(data)


if __name__ == '__main__':
    data = loadMetaData()
    for i in tqdm.tqdm(range(1, NUMFOLDS + 1)):
        shuffledData = data
        np.random.shuffle(shuffledData)  # Data is now shuffled!
        length = len(data)
        reduction_perc = int(length * TESTPERC)
        val_start = length - (2 * reduction_perc)
        val_end = length - reduction_perc
        directory = "../CameraPrimusFolds/Fold" + str(i)
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        vector_to_file(directory + "/train", data[:val_start])  # Train dataset
        vector_to_file(directory + "/validation", data[val_start:val_end])  # Validation train dataset
        vector_to_file(directory + "/test", data[val_end:])  # Test dataset





