import pyjapc
import os
import time
import dill as pickle
import zstandard as zstd
import datetime as dt
import numpy as np
from scipy.optimize import curve_fit

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
dctx = zstd.ZstdDecompressor()

DATA_PATH_BWS = '/user/jpucek/BWS/Data'
DATA_PATH_BSRH = '/user/jpucek/BSRH/Data'

#Open last file of each BWS:
directoryH = os.path.join(DATA_PATH_BWS+'/Horizontal', time.strftime('%Y/%m/%d/', time.localtime()))
filesH = os.listdir(directoryH)
filesH = sorted(filesH)

directoryV = os.path.join(DATA_PATH_BWS+'/Vertical', time.strftime('%Y/%m/%d/', time.localtime()))
filesV = os.listdir(directoryV)
filesV = sorted(filesV)

directoryBSRH = os.path.join(DATA_PATH_BSRH, time.strftime('%Y/%m/%d/', time.localtime()))
filesBSRH = os.listdir(directoryBSRH)
filesBSRH = sorted(filesBSRH)

#Let's start with horizontal BWS:
f = open(directoryH + filesH[-1], 'rb') #Open last file with the latest BWS profile...
with dctx.stream_reader(f) as reader:
    BWS_H = pickle.load(reader)

print('BWS_H loaded')
g = open(directoryH + filesV[-1], 'rb')
with dctx.stream_reader(g) as reader:
    BWS_V = pickle.load(reader)
print('BWS_V loaded')
##### So we have the BWS data, let's find the corresponding BSRH - we want one image for H and one image for V:

timeH = BWS_H.header['acqStamp']
timeV = BWS_V.header['acqStamp']

nFiles = len(filesBSRH)