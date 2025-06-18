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
if filesH == None:
    print('There are no horizontal BWS today...')
else:
    filesH = sorted(filesH)
    # Let's start with horizontal BWS:
    f = open(directoryH + filesH[-1], 'rb')  # Open last file with the latest BWS profile...
    with dctx.stream_reader(f) as reader:
        BWS_H = pickle.load(reader)
    print('BWS_H loaded')

directoryV = os.path.join(DATA_PATH_BWS+'/Vertical', time.strftime('%Y/%m/%d/', time.localtime()))
filesV = os.listdir(directoryV)
if filesV == None:
    print('There are no horizontal BWS today...')
else:
    filesV = sorted(filesV)
    g = open(directoryV + filesV[-1], 'rb')
    with dctx.stream_reader(g) as reader:
        BWS_V = pickle.load(reader)
    print('BWS_V loaded')


directoryBSRH = os.path.join(DATA_PATH_BSRH, time.strftime('%Y/%m/%d/', time.localtime()))
filesBSRH = os.listdir(directoryBSRH)
filesBSRH = sorted(filesBSRH)
##### So we have the BWS data, let's find the corresponding BSRH - we want one image for H and one image for V:

timeH = dt.datetime.fromtimestamp(BWS_H.value['gmtTimestampAcquireIn_ns']/1e9) #Or use 'gmtTimestampAcquireIn_ns'
timeV = dt.datetime.fromtimestamp(BWS_V.value['gmtTimestampAcquireIn_ns']/1e9)
print('BWS Acq Time H: ', timeH)
print('BWS Acq Time V: ', timeV)

nFiles = len(filesBSRH)
h = open(directoryBSRH+filesBSRH[nFiles-1], 'rb') #Open last known file
with dctx.stream_reader(h) as reader:
    BSRH = pickle.load(reader)

#Now get time of first and last BSRH event:
tFirstEvent = dt.datetime.fromtimestamp(BSRH[0].cam_data['imageTimeStamp']/1e9)
tLastEvent = dt.datetime.fromtimestamp(BSRH[-1].cam_data['imageTimeStamp']/1e9)
print('First event in the BSRH file: ', tFirstEvent)
print('Last event in the BSRH file:', tLastEvent)

#So let's see whether the time of BWS is within our file: