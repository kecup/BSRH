import datetime as dt
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass
from time import sleep
from scipy.signal import find_peaks
from bi_pyutils import rda

def gaussian(x, amp, cen, wid, const):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2)) + const

def lowLength(array):
    #Identify maxima from a moving average, to smear the hot pixels
    peaks, properties = find_peaks(array, height=300, distance=10, width=5)

    if len(peaks) < 2:
        # Not enough peaks found — return 0
        return 0

    # Get the corresponding peak values
    values = np.array(array)[peaks]

    # Find indices of the two highest peaks
    top_two_indices = peaks[np.argsort(values)[-2:][::-1]]

    # Return distance between them
    return abs(top_two_indices[1] - top_two_indices[0])

class Pneumatics:
    def __init__(self, name, japc, slot):
        self.name = name
        self.slot = slot
        self.japc = japc
        self.state = []
        #self.getPosition()

    def getPosition(self): # This will have to provide the full channel state that will be used to move the targets...
        state = self.japc.getParam(self.name+'/Setting#channelState')
        self.state = np.asarray(state)
        return state[self.slot]

    def isIn(self):
        return self.getPosition()

    def changePosition(self): #If in we invert to out, if out we change to in...
        position = self.getPosition()

        if position == 1:
            self.state[self.slot] = 0
        else:
            self.state[self.slot] = 1
        self.japc.setParam(self.name+'/Setting#channelState', self.state, checkDims=False)

    def isBusy(self): #This is just experimental!
        while np.any(self.japc.getParam(self.name+'/Acquisition#channelState') != self.state):
            sleep(3)
        return False

    def moveIn(self):
        self.getPosition()
        self.state[self.slot] = 1
        self.japc.setParam(self.name + '/Setting#channelState', self.state, checkDims=False)

    def moveOut(self):
        self.getPosition()
        self.state[self.slot] = 0
        self.japc.setParam(self.name + '/Setting#channelState', self.state, checkDims=False)

## Few comments on the motor logic - a) motion is non-blocking, so one has to carefully treat the busy statement, b) busy does not get turned on the milisecond one sends a new position to go to, c) all motion of a motor is done in raw steps
## d) getPosition is blocking, if non-blocking position is required, use Motor.position
class Motor:
    def __init__(self, name, japc, backlash_steps=0, inPos = None, outPos = None): ##inPos/outPos should be provided in case one wants to use the changePosition function
        self.name = name
        self.position = -1
        self.desiredAbsPosition = -1
        self.inPos = inPos
        self.outPos = outPos
        self.busy = True
        self.lastMotionRel = 0
        self.lastMotionAbs = 0
        self.backlash_steps = backlash_steps
        self.japc = japc
        self.japc.subscribeParam(self.name+'/Acquisition#position_counter', self.callbackPosition)
        self.japc.subscribeParam(self.name + '/Acquisition#motor_busy', self.callbackBusy)
        # try:
        #     if self.japc.getParam(self.name + '/Acquisition#motor_busy') == False:
        #         self.position = self.japc.getParam(
        #             self.name + '/Acquisition#position')  # This is just to test that acquisition can be made
        #     else:
        #         print("Motor is moving!")
        # except:
        #     print('Invalid name!')

    def isIn(self):
        if self.inPos == self.position:
            return 1
        else:
            return 0

    def isBusy(self, debug = False):
        if debug == True:
            print(self.position, self.desiredAbsPosition)

        return self.busy

    def callbackPosition(self, name, value):
        self.position = value

    def callbackBusy(self, name, value):
        self.limitSwitchActive()
        if self.position == self.desiredAbsPosition:
            self.busy = value
        else:
            self.busy = True

    def getPosition(self):  ## This is NEW
        while self.japc.getParam(self.name + '/Acquisition#motor_busy') == True:
            sleep(0.5)
        self.position = self.japc.getParam(
            self.name + '/Acquisition#position_counter')  # Uses position in steps! So carefull with movements...
        return self.position

    def limitSwitchActive(self):
        plusSwitch = self.japc.getParam(self.name + '/Acquisition#end_switch_plus_active')
        minusSwitch = self.japc.getParam(self.name + '/Acquisition#end_switch_minus_active')
        out = -1

        if plusSwitch:
            out = True
            if self.desiredAbsPosition > self.position:
                self.desiredAbsPosition = self.position
                self.busy = False

        elif minusSwitch:
            out = True
            if self.desiredAbsPosition < self.position:
                self.desiredAbsPosition = self.position

        else:
            out = False
        return out

    def setAbsPosition(self, absPosition):
        absPosition = int(absPosition)
        self.desiredAbsPosition = absPosition

        if self.japc.getParam(self.name + '/Acquisition#motor_busy') == False:

            if np.abs(absPosition-self.position) > self.backlash_steps: # when change bigger than backlash, only direction needed
                self.lastMotionRel = absPosition-self.position
            elif self.lastMotionRel * (absPosition-self.position) < 0: # this is when the direction is changed only partially - need to remember the amount of steps; the other scenario is by default ok..
                self.lastMotionAbs += (absPosition-self.position) # do not change the backlash direction yet..
            mv = {'value': absPosition, 'mode': 2, 'units': 2}
            self.japc.setParam(self.name + '/Move', mv)
            self.busy = True
            #while absPosition != self.position:
            #    sleep(0.5)

        else:
            print(f'Cannot comply, {self.name} is busy!')

    def setRelPosition(self, relPosition):  # Motion in relative steps!
        relPosition = int(relPosition)
        if self.japc.getParam(self.name + '/Acquisition#motor_busy') == False:

            if relPosition * self.lastMotionRel < 0:  # Change of sign
                self.lastMotionRel = relPosition
                relPosition = relPosition + np.sign(relPosition) * self.backlash_steps - self.lastMotionAbs
            else:
                self.lastMotionRel = relPosition - self.lastMotionAbs
            if self.lastMotionAbs != 0:
                self.lastMotionAbs = 0

            self.desiredAbsPosition = self.position+relPosition
            mv = {'value': relPosition, 'mode': 1, 'units': 2}
            self.japc.setParam(self.name + '/Move', mv)
            self.busy = True

        else:
            print('Cannot comply, motor is busy!')

    def changePosition(self):
        if self.inPos is not None and self.outPos is not None: #Logic is - if out send in, if anything else, send out...
            if self.outPos == self.position: #Send in
                self.setAbsPosition(self.inPos)
            else:#Send out
                self.setAbsPosition(self.outPos)
        else:
            print('In/out not provided, doing nothing')

class FWheel(Motor): # This is a motor with specific positions/values
    ## Redefining the motion, using units of filters...

    def getPosition(self):  ## This is NEW
        while self.japc.getParam(self.name + '/Acquisition#motor_busy') == True:
            sleep(0.5)
        return np.round(self.japc.getParam(self.name + '/Acquisition#position')) # Uses position in units! So careful with movements...

    def setAbsPosition(self, absPosition):
        absPosition = int(absPosition)
        self.desiredAbsPosition = absPosition

        if self.japc.getParam(self.name + '/Acquisition#motor_busy') == False:

            if np.abs(absPosition-self.position) > self.backlash_steps: # when change bigger than backlash, only direction needed
                self.lastMotionRel = absPosition-self.position
            elif self.lastMotionRel * (absPosition-self.position) < 0: # this is when the direction is changed only partially - need to remember the amount of steps; the other scenario is by default ok..
                self.lastMotionAbs += (absPosition-self.position) # do not change the backlash direction yet..
            mv = {'value': absPosition, 'mode': 2, 'units': 1} #In the units of the filter wheel
            self.japc.setParam(self.name + '/Move', mv)
            self.busy = True
            #while absPosition != self.position:
            #    sleep(0.5)

        else:
            print('Cannot comply, motor is busy!')

    def setRelPosition(self, relPosition):  # Motion in relative steps!
        relPosition = int(relPosition)
        if self.japc.getParam(self.name + '/Acquisition#motor_busy') == False:

            if relPosition * self.lastMotionRel < 0:  # Change of sign
                self.lastMotionRel = relPosition
                relPosition = relPosition + np.sign(relPosition) * self.backlash_steps - self.lastMotionAbs
            else:
                self.lastMotionRel = relPosition - self.lastMotionAbs
            if self.lastMotionAbs != 0:
                self.lastMotionAbs = 0

            self.desiredAbsPosition = self.position+relPosition
            mv = {'value': relPosition, 'mode': 1, 'units': 1} #units of the filter wheel
            self.japc.setParam(self.name + '/Move', mv)
            self.busy = True

        else:
            print('Cannot comply, motor is busy!')

class Camera:
    def __init__(self, name, japc, remDevAcc = True):  ## Provide motors as Class and positions in steps...
        self.name = name
        self.japc = japc
        self.updated = dt.datetime.now()
        self.image = []
        self.bgImage = []
        self.remDevAcc = remDevAcc
        if self.remDevAcc:
            self.rda = rda.RDA()
            self.rda.init(kerberos = True)
            self.exposure = self.japc.getParam(self.name + '/Acquisition#ExposureTime')
            self.subscription = self.japc.subscribeParam(self.name + '/Acquisition#image1', self.updateImage)
        else:
            self.exposure = self.japc.getParam(self.name + '/CameraSetting#cameraExposureTimeUs')
            self.subscription = self.japc.subscribeParam(self.name + '/LastImage#image2D', self.updateImage) # This is the fast subscription, should not run it unless necessary!!

    def changeExpTime(self, exposure):
        exposure = int(np.round(exposure))
        if 40 <= exposure <= 2e5:
            self.exposure = exposure
            print(f'Setting {self.name} exposure to: {exposure}')
        elif exposure < 40:
            self.exposure = 40
            print('Exposure not valid, setting min = 40us')
        elif exposure > 2e5:
            self.exposure = 2e5
            print('Exposure not valid, setting max = 2e5us') # TODO: for cam4 and 6 there should be a way to lower the amount of filters


        if self.remDevAcc:
            try:
                self.rda.set(self.name,'Setting',[{"field": 'ExposureTime', "type": rda.RDA_INT, "data": self.exposure}])
            except:
                print('Setting exposure failed')
        else:
            self.japc.setParam(self.name+'/CameraSetting#cameraExposureTimeUs', self.exposure)

    def setBg(self, image): # We should make some fail-safes...
        self.bgImage = image

    def updateImage(self, parameter, value):
        self.image = value
        self.updated = dt.datetime.now()

    def captureImage(self, bg = False):  ## Should think whether subscription is the way forward or not...
        lastUpdate = self.updated
        oldImage = self.image
        cycle = 0
        while lastUpdate == self.updated and cycle < 10:
            sleep(0.2)
            cycle = cycle + 1

        if cycle >= 10:
            print('Getting image timed-out.')
            outImage = np.asarray([])
        else:
            if ((oldImage == self.image).all()):
                print('The same old image!!!') # TODO: In this case we need to powercycle...
                outImage = np.asarray([])
            else:
                outImage = self.image

        if bg:
            self.bgImage = outImage
        return outImage

    def getBeamPosition(self, com = False, occulter = None): #com = Center of Mass, different way of center determination for images that cannot be represented by a Gaussian..., occulter provides list of x,y that is the supposed center of the occulter
        ##Get an image that we will analyse:
        try:
            if len(self.bgImage) != 0:
                image = np.asarray(self.captureImage()) - self.bgImage # Well, here we should be careful that the exposure time did not change
            else:
                image = np.asarray(self.captureImage())

        except:
            print('Problem capturing image!')
            return -1,-1
        outX = -1
        outY = -1

        xTemp = range(len(image[0]))
        yTemp = range(len(image))

        if occulter != None: # What if we provide occulter but it is not there???
            lengthX = []
            lengthY = []

            # Refine the location
            # for offset in range(-5,6): #Well, this might be problematic in case the occulter is not there, yet we perform this...
            #     crosscutX = image[occulter[0]+offset,:]
            #     crosscutY = np.transpose(image[:,occulter[1]+offset])
            #     lengthX.append(lowLength(crosscutX))
            #     lengthY.append(lowLength(crosscutY))
            #
            # # Here we use the one with the longest low, should be the same in both axis..:
            # Radius = (np.max(lengthX) + np.max(lengthY))/2 + 5
            # print(lengthX)
            # print(lengthY)
            #
            # if Radius < 15:
            #     print('No occulter detected')
            # else:
            #     OCX = occulter[0] + lengthX.index(np.max(lengthX))
            #     OCY = occulter[1] + lengthY.index(np.max(lengthY))
            #
            #     xTemp = list(range(0, int(OCX-Radius)))+list(range(int(OCX+Radius), len(image[0])-1))
            #     yTemp = list(range(0, int(OCY-Radius)))+list(range(int(OCY+Radius), len(image)-1))

            xTemp = list(range(0, int(occulter[0] - occulter[2]))) + list(range(int(occulter[0] + occulter[2]), len(image[0]) - 1))
            yTemp = list(range(0, int(occulter[1] - occulter[2])))+list(range(int(occulter[1] + occulter[2]), len(image)-1))


        if com == True:
            center = center_of_mass(image)
            outX = center[1]
            outY = center[0]

        else:
            try:
                projection_x = np.sum(image, axis=0)
                projection_y = np.sum(image, axis=1)
            except:
                projection_x = []
                projection_y = []

            ##Prepare bounds for fits:
            gBounds = ([1e2, 0, 3, 10], [1e7, 960, 100, 2e4])

            try:
                # resX = [np.max(projection_x), np.argmax(projection_x)]
                resX, _ = curve_fit(gaussian, xTemp, projection_x[xTemp],
                                    (6e4, np.argmax(projection_x), 10.0, 300.0), bounds=gBounds)
                if resX[0] < 2e3:
                    resX[1] = -1
            except:
                print("Did not fit in X.")
                resX = [-1, -1, -1]

            try:
                # resY = [np.max(projection_y), np.argmax(projection_y)]
                resY, _ = curve_fit(gaussian, yTemp, projection_y[yTemp],
                                    (6e4, np.argmax(projection_y), 10.0, 300.0), bounds=gBounds)
                if resY[0] < 2e3:
                    resY[1] = -1
            except:
                print("Did not fit in Y.")
                resY = [-1, -1, -1]

            outX = resX[1]
            outY = resY[1]

        return outX, outY

    def deleteBG(self):
        self.bgImage = []

    def getBeamInfo(self):
      """
      :return: resX, resY ([amplitude, center, sigma])
      """
      ##Get an image that we will analyse:
      image = self.captureImage()
      ##Perform a 1D fit of projections:
      try:
          projection_x = np.sum(image, axis=0)
          projection_y = np.sum(image, axis=1)
      except:
          projection_x = []
          projection_y = []

      ##Prepare bounds for fits:
      gBounds = ([1000, 0, 1, 10], [1e5, 960, 30, 1e3])

      try:
          # resX = [np.max(projection_x), np.argmax(projection_x)]
          resX, _ = curve_fit(gaussian, range(len(projection_x)), projection_x,
                              (6e4, np.argmax(projection_x), 10.0, 300.0), bounds=gBounds)
          if resX[0] < 2e4:
              resX[1] = -1
      except:
          print("Did not fit in X.")
          resX = [-1, -1, -1]

      try:
          # resY = [np.max(projection_y), np.argmax(projection_y)]
          resY, _ = curve_fit(gaussian, range(len(projection_y)), projection_y,
                              (6e4, np.argmax(projection_y), 10.0, 300.0), bounds=gBounds)
          if resY[0] < 2e4:
              resY[1] = -1
      except:
          print("Did not fit in Y.")
          resY = [-1, -1, -1]

      return resX, resY