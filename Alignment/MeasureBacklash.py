import numpy as np
import pyjapc
import math
import time
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from devices_lib import Camera, Motor
import pyrbac

def distance(p1, p2):
    """
    Return Euclidean distance between two 2D points (a, b).
    Returns -1 if input is invalid.
    """
    try:
        # Ensure input is iterable with exactly 2 numeric elements
        if (
            not all(isinstance(p, (list, tuple)) for p in (p1, p2)) or
            len(p1) != 2 or len(p2) != 2 or
            not all(isinstance(coord, (int, float)) for coord in (*p1, *p2))
        ):
            raise ValueError("Invalid input")

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.hypot(dx, dy)

    except Exception:
        return -1

########Parameters and names#################

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
client = pyrbac.AuthenticationClient()
token = client.login_kerberos()
japc.rbacLogin(token=token)

##We should have most components ready, let's get data for training:
##BackLash
FoldingX = Motor('LHC.BSRH.M3-PHI.B2', japc) # todo
# FoldingX = Motor('LHC.BSRH.M3-THETA.B2', japc)
cam = Camera('LHC.BSRH.BTVDC-CAM.MAIN', japc, False)

motion = 130 # steps

beamPosStart = []

japc.startSubscriptions()
time.sleep(1)

for i in range(10):
    beamPosStart.append(cam.getBeamPosition())
averagePos = np.mean(beamPosStart, axis = 0)
var = np.std(beamPosStart, axis = 0)

print('Starting')
print('AvgX = ', averagePos[0], '+- ', var[0])
print('AvgY = ', averagePos[1], '+- ', var[1])

stepsTaken = 0

# Move forward until motion is detected....
while -1 < distance(cam.getBeamPosition(), list(averagePos)) < 5*np.sqrt(var[0]**2 + var[1]**2):
    FoldingX.setRelPosition(5)
    stepsTaken+=1
    time.sleep(0.2)

print('Steps until motion = ', stepsTaken)
stepsTaken = 0

beamPosMotion = []
for i in range(10):
    beamPosMotion.append(cam.getBeamPosition())
averageMotion = np.mean(beamPosMotion, axis = 0)
varMotion = np.std(beamPosMotion, axis = 0)

FoldingX.setRelPosition(motion) # Then calculate distance for those 100 steps
time.sleep(2)

beamPosStart = []
for i in range(10):
    beamPosStart.append(cam.getBeamPosition())
averagePos = np.mean(beamPosStart, axis = 0)
var = np.std(beamPosStart, axis = 0)

pxForward = distance(list(averagePos), list(averageMotion))
print('Distance = ', pxForward)
## Now we go back 100 steps to see the change in position on camera and we'll see how many steps we've lost

FoldingX.setRelPosition(-motion)
time.sleep(2)

beamPosMotion = []
for i in range(10):
    beamPosMotion.append(cam.getBeamPosition())
averageMotion = np.mean(beamPosMotion, axis = 0)
varMotion = np.std(beamPosMotion, axis = 0)

pxBackward = distance(list(averagePos), list(averageMotion))
print('Distance = ', pxBackward)

print('Number of backlash steps is: ', (1-pxBackward/pxForward)*motion)


# print('AvgX = ', averagePos[0], '+- ', var[0])
# print('AvgY = ', averagePos[1], '+- ', var[1])
# #Then we move backward until we see motion...
# while -1 < distance(cam.getBeamPosition(), list(averagePos)) < 3*np.sqrt(var[0]**2 + var[1]**2):
#     FoldingX.setRelPosition(-5)
#     time.sleep(0.2)
#     stepsTaken +=1
# print('Steps until motion = ', stepsTaken)

japc.stopSubscriptions()
japc.clearSubscriptions()