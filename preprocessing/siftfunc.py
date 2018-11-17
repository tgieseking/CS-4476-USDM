import cv2
import numpy as np
from getSift import getSift
from tempfile import TemporaryFile
import os
# import 'input_paths.npy' as inputArr
# import 'output_paths.npy' as outputArr

inArr = np.load('input_paths.npy')
outArr = np.load('output_paths.npy')
for i in range (0, len(inArr)):
    curIn = inArr[i]
    curIn = os.getcwd()+'/'+curIn
    img = cv2.imread(curIn)
    siftArr = getSift(img)
    curOut = outArr[i]
    curOut = os.getcwd()+'/'+curOut
    np.save(curOut, siftArr)
# outfile = TemporaryFile()
# img = cv2.imread(input)
# siftArr = getSift(img)
# np.save('output.npy', siftArr)
