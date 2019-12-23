# ----------------------------------------------------- #
# Project   :   Poisson Image Editing                   #
# Title     :   main.py                                 #
# Detail    :   Load image and save adapted results     #
#                                                       #
# Author    :   Xu Ruihan                               #
# ID        :   1700017793                              #
# ----------------------------------------------------- #

# input image
import os
import errno
from glob import glob
import cv2
# paint and move mask
# import paint_mask
# import move_mask
# poisson editing

from poisson import Poisson

IMG_EXTENSIONS = [
    'png', 'jpeg', 'jpg', 'gif', 'tiff', 'tif', 'raw', 'bmp'
]
SRC = 'SRC'
RES = 'RES'


def collectFiles(prefix, extension_list=IMG_EXTENSIONS):
    FileNames = sum(map(glob, [prefix + ext for ext in extension_list]), [])
    return FileNames


if __name__ == '__main__':
    InputFolders = list(os.walk(SRC))
    InputFolders.pop(0)

    for parent, dirnames, filenames in InputFolders:
        InputDir = os.path.split(parent)[-1]
        OutputDir = os.path.join(RES, InputDir)
        print(f"Processing input {InputDir}...")

        SourceFile = collectFiles(os.path.join(parent, '*source.'))
        TargetFile = collectFiles(os.path.join(parent, '*target.'))
        MaskFile = collectFiles(os.path.join(parent, '*mask.'))

        if len(SourceFile) != 1 or len(TargetFile) != 1 or len(MaskFile) != 1:
            print("There must be one source, one target and one mask. "
                  "Please check the input files")
            continue

        # Input images
        source = cv2.imread(SourceFile[0], cv2.IMREAD_COLOR)
        target = cv2.imread(TargetFile[0], cv2.IMREAD_COLOR)
        mask = cv2.imread(MaskFile[0], cv2.IMREAD_COLOR)

        PoissonEditor = Poisson(source, target, mask)
        print(f'Input files in {InputDir}. Editing...')
        result = PoissonEditor.edit(setting=InputDir)

        # Make directory for results
        try:
            os.makedirs(OutputDir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        cv2.imwrite(os.path.join(OutputDir, 'result.png'), result)
        print(f"Finished processing {InputDir}")
