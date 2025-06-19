import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import math
from scipy import fft
from scipy.stats import skewnorm
from scipy.stats import exponnorm
from scipy.optimize import curve_fit
import os
import math
import lmfit
import dill as pickle
import pyjapc
import zstandard as zstd

### This script opens the data acquired by the BSRH (+ additional devices), sample to be change accordingly
#### Run in interactive mode...

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='') #Stupid Java VM must be running to open the java values... We have to retype this...

dctx = zstd.ZstdDecompressor()
directory = '/user/jpucek/BSRH/Data/2025/06/19/'
files = os.listdir(directory)
files = sorted(files)

profilesV = []
profilesH = []
bunchN = []
getBunchN = []
imageSum = []

for nFile in range(len(files)-1, len(files)):
    f = open(directory + files[nFile], 'rb')
    with dctx.stream_reader(f) as reader:
        a = pickle.load(reader)

    for i in range(len(a)):
        bunchN.append(a[i].device_data['GATE-INT']['delaySlots'])
        imageSum.append(np.sum(a[i].cam_data['image2D']))

#plt.ion()
plt.plot(imageSum)
plt.figure(2)
plt.plot(bunchN)
plt.show()