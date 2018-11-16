import cv2
import numpy as np
from getSift import getSift
from tempfile import TemporaryFile

def siftFunc(input, output):
    outfile = TemporaryFile()
    img = cv2.imread('input.jpg')
    siftArr = getSift(img)
    np.save('output.npy', siftArr)
