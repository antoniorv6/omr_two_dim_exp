##Loads Camera Primus Dataset from a fold
def LoadCameraPrimus(filepath, samples):
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


def prepareOutput1(trainY, testY, valY, i2w, w2i, manuscript):

    output_sos = '<s>'
    output_eos = '</s>'

    Y_train = [[output_sos] + sequence + [output_eos] for sequence in trainY]
    Y_test = [[output_sos] + sequence + [output_eos] for sequence in testY]
    Y_val = [[output_sos] + sequence + [output_eos] for sequence in valY]

    # Setting up the vocabulary with positions and symbols

    w2i, i2w, LENGTH = confectVocabulary(Y_train, Y_test, Y_val, w2i, i2w, manuscript, 1)

    return Y_train, Y_test, Y_val, w2i, i2w, LENGTH

def prepareOutput3(trainY, testY, valY, i2w, w2i, manuscript):

    output_sos = '<s>'
    output_eos = '</s>'

    Y_train = [[output_sos] + parseSequence(sequence) + [output_eos] for sequence in trainY]
    Y_test = [[output_sos] + parseSequence(sequence) + [output_eos] for sequence in testY]
    Y_val = [[output_sos] + parseSequence(sequence) + [output_eos] for sequence in valY]

    # Setting up the vocabulary with positions and symbols

    w2i, i2w, LENGTH = confectVocabulary(Y_train, Y_test, Y_val, w2i, i2w, manuscript, 3)

    return Y_train, Y_test, Y_val, w2i, i2w, LENGTH

def confectVocabulary(YT, YTest, YVal, w2i, i2w, manuscript, type_voc):
    
    vocabulary = set()

    for sequence in YT:
        vocabulary.update(sequence)
    for sequence in YTest:
        vocabulary.update(sequence)
    for sequence in YVal:
        vocabulary.update(sequence)

    ALPHABETLENGTH = len(vocabulary) + 1

    w2i = dict([(char, i+1) for i, char in enumerate(vocabulary)])
    i2w = dict([(i+1, char) for i, char in enumerate(vocabulary)])

    w2i['PAD'] = 0
    i2w[0] = 'PAD'

    np.save("vocabulary/" + manuscript + "/w2i" + str(type_voc) + ".npy", w2i)
    np.save("vocabulary/"+ manuscript + "/i2w"  + str(type_voc) + ".npy", i2w)

    return w2i, i2w, ALPHABETLENGTH

def resize_image(FEATURESPERFRAME, image):
    resize_width = int(float(FEATURESPERFRAME * image.shape[1]) / image.shape[0])
    return cv2.resize(image, (resize_width, FEATURESPERFRAME))

def parseSequence(sequence):
    parsed = []
    for char in sequence:
        parsed += char.split("-")
    return parsed