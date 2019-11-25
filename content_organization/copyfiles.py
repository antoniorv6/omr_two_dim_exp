import os
import os.path
import shutil


def putInList(imagename, ispng):
    imagename_no_extension = imagename.split('.')[0]
    if not ispng:
        imagename_no_extension = '_'.join(imagename_no_extension.split('_')[:-1])

    agnosticname_no_extension = agnosticname.split('.')[0]

    if imagename_no_extension == agnosticname_no_extension:
        imagepath = "CameraPrimus/" + imagename
        agnosticpath = "CameraPrimus/" + agnosticname
        finalList.write("%s\t%s\n" % (imagepath, agnosticpath))


copyDirectory = "../CameraPrimus"
agnosticname = ""
imagepath = ""

finalList = open("dataset.lst", "w+")

for dirpath, dirnames, filenames in os.walk("./Corpus"):
    for filename in [f for f in filenames]:
        if filename.endswith(".agnostic"):
            agnosticpath = os.path.join(dirpath, filename)
            shutil.copy(agnosticpath, copyDirectory)
            agnosticname = filename

        if filename.endswith(".png"):
            pngsplitlen = len(filename.split('.'))
            if (pngsplitlen < 3):
                pngpath = os.path.join(dirpath, filename)
                shutil.copy(pngpath, copyDirectory)
                putInList(filename, True)

        if filename.endswith(".jpg"):
            imagepath = os.path.join(dirpath, filename)
            shutil.copy(imagepath, copyDirectory)
            putInList(filename, False)

finalList.close()
