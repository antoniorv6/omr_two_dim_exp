
import tqdm
import numpy as np
import os
import sys

import json

# ==== CONSTANTS ====
NUMFOLDS = 5
TESTPERC = 0.2
VALIDATIONPERC = 0.2
# ===================


def vector_to_file(filename, data):
    with open(filename, "w") as file:
        file.writelines(data)
    file.close()

def loadData(path):
    fileList = []

    valid = False
    for file in os.listdir(path):
        if file.endswith(".json"):
            with open(path + file) as json_file:
                data = json.load(json_file)
                for page in data['pages']:
                    if "regions" in page:
                        for region in page['regions']:
                            if region['type'] == 'staff' and "symbols" in region:
                                fileList.append(path+file + ' \n')
                                valid = False
                                break
                    if valid == True:
                        break

    return fileList


if __name__ == '__main__':
    data = loadData("dataHandwritten/B-59.850/")
    
    for i in tqdm.tqdm(range(1, NUMFOLDS + 1)):
        shuffledData = data
        np.random.shuffle(shuffledData)  # Data is now shuffled!
        length = len(data)
        reduction_perc = int(length * TESTPERC)
        val_start = length - (2 * reduction_perc)
        val_end = length - reduction_perc
        directory = "HandwrittenFolds/Fold" + str(i)
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        vector_to_file(directory + "/train", shuffledData[:val_start])  # Train dataset
        vector_to_file(directory + "/validation", shuffledData[val_start:val_end])  # Validation train dataset
        vector_to_file(directory + "/test", shuffledData[val_end:])  # Test dataset





