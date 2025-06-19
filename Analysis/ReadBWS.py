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
import datetime as dt

### This script opens the data acquired by the BWS, sample to be change accordingly
#### Run in interactive mode...

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='') #Stupid Java VM must be running to open the java values... We have to retype this...

dctx = zstd.ZstdDecompressor()
directory = '/user/jpucek/BWS/Data/Vertical/2025/06/19/'
files = os.listdir(directory)
files = sorted(files)

sigma = 4.7
verSize = 386
#horSize = 257
bunchStart = 160
bunchStop = 800

now = dt.datetime.now()
start = dt.datetime.combine(now.date(),dt.time(15,11,54))
stop = dt.datetime.combine(now.date(),dt.time(15,13,7))

for nFile in range(0, len(files)):
    f = open(directory + files[nFile], 'rb')
    with dctx.stream_reader(f) as reader:
        a = pickle.load(reader)
    if start <= dt.datetime.fromtimestamp(a.value['gmtTimestampAcquireIn_ns']/1e9)<= stop:
        print(nFile)

