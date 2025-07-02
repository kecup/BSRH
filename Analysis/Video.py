import dill as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import pyjapc
import zstandard as zstd

dctx = zstd.ZstdDecompressor()
directory = '/user/jpucek/BSRH/Data/2025/06/19/'
files = os.listdir(directory)
files = sorted(files)

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')

imArr = []
intensity = []

fig, ax = plt.subplots()

bunch = 160

for i in range(80, len(files)):

    f = open(directory + files[i], 'rb')
    with dctx.stream_reader(f) as reader:
        a = pickle.load(reader)

    for j in range(len(a)):
        if a[j].device_data['GATE-INT']['delaySlots'] == bunch and a[j].device_data['GATE-INT']['pulseLength'] <= 10:
            image = a[j].cam_data['image2D']
            image = image - np.mean(image[:100, :100])  # Test n1
            imArr.append(image)

            intensity.append(np.sum(image[180:480, 310:610]))

    # time = a.cam_data['acqStamp']
    # BSRH_time = time+dt.timedelta(hours=2)
    # BSRH_time = BSRH_time.replace(tzinfo = None)
    # timeArr.append(BSRH_time)

    # image = image - np.mean(image[:50,:]) #Test n2
    # projY.append(np.sum(image[:,460:470],axis = 1))
    f.close()
# plt.plot(timeArr, intensity)

tst = plt.imshow(imArr[0], 'jet', vmax=3000, vmin=0)
plt.colorbar()


def update(frame):
    tst.set_data(imArr[frame])  # -imArr[0])
    return (tst)


ani = animation.FuncAnimation(fig=fig, func=update, frames=len(imArr), interval=100, repeat=False)

writer = animation.PillowWriter(fps=20, metadata=dict(artist='Me'))
ani.save('Video_{}.gif'.format(bunch), writer=writer)

# plt.show()