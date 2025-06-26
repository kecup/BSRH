import datetime as dt
import numpy as np
from scipy.optimize import curve_fit
from time import sleep


def gaussian(x, amp, cen, wid, const):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2)) + const

class Pneumatics:
    def __init__(self, name, japc, slot):
        self.name = name
        self.slot = slot
        self.japc = japc

    def getPosition(self): # This will have to provide the full channel state that will be used to move the targets...
        state = self.japc.getParam(self.name+'/Setting#channelState')
        return state[self.slot]

    def changePosition(self): #If in we invert to out, if out we change to in...
        state = self.getPosition()

        if state[self.slot] == 1:
            state[self.slot] = 0
        else:
            state[self.slot] = 1
        self.japc.setParam(self.name+'/Setting#channelState', state)

class Motor:
    def __init__(self, name, japc, backlash_steps=0):
        self.name = name
        self.position = -1
        self.desiredAbsPosition = -1
        self.busy = True
        self.lastMotionRel = 0
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

    def isBusy(self):
        return self.busy

    def callbackPosition(self, name, value):
        self.position = value

    def callbackBusy(self, name, value):
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

    def setAbsPosition(self, absPosition):  # This is motion in absolute raw steps! No backlash taken into account!!!
        absPosition = int(absPosition)
        self.desiredAbsPosition = absPosition
        if self.japc.getParam(self.name + '/Acquisition#motor_busy') == False:
            mv = {'value': absPosition, 'mode': 2, 'units': 2}
            self.japc.setParam(self.name + '/Move', mv)
            self.busy = True
            #while absPosition != self.position:
            #    sleep(0.5)

        else:
            print('Cannot comply, motor is busy!')

    def setRelPosition(self, relPosition):  # Motion in relative steps!
        if self.japc.getParam(self.name + '/Acquisition#motor_busy') == False:

            if relPosition * self.lastMotionRel < 0:  # Change of sign
                self.lastMotionRel = relPosition
                relPosition = relPosition + np.sign(relPosition) * self.backlash_steps
            else:
                self.lastMotionRel = relPosition

            mv = {'value': relPosition, 'mode': 1, 'units': 2}
            self.japc.setParam(self.name + '/Move', mv)

        else:
            print('Cannot comply, motor is busy!')


class Camera:
    def __init__(self, name, japc, inPos=None, outPos=None, motorX=None,
                 motorY=None):  ## Provide motors as Class and positions in steps...
        self.name = name
        self.motorX = motorX
        self.motorY = motorY
        self.japc = japc
        self.inPos = inPos
        self.outPos = outPos
        self.updated = dt.datetime.now()
        self.image = []
        if inPos != None and outPos != None:
            self.movableAxis = len(inPos)
        else:
            self.movableAxis = 0
        self.isIn = -1
        try:
            self.subscription = self.japc.subscribeParam(self.name + '/Acquisition#image1', self.updateImage)
        except:
            self.subscription = self.japc.subscribeParam(self.name + '/LastImage#image2D', self.updateImage)

    def updateImage(self, parameter, value):
        self.image = value
        self.updated = dt.datetime.now()

    def checkIn(self):
        output = False
        if (self.movableAxis > 0 and motorX.getPosition() == inPos[0]) and (
                self.movableAxis == 1 or motorY.getPosition() == inPos[1]) or self.movableAxis == 0:
            output = True
        return output

    def captureImage(self, MoveOut=False):  ## Should think whether subscription is the way forward or not...
        lastUpdate = self.updated
        if self.checkIn() == False:  # Checking if the camera/target is in.
            ##This means that we have to move motor(s)
            motorX.setAbsPosition(inPos[0])  # at least one motor must be present
            if movableAxis == 2:
                motorY.setAbsPosition(inPos[1])
                while motorX.isBusy() or motorY.isBusy():
                    sleep(0.2)
            while motorX.isBusy():
                sleep(0.2)

        cycle = 0
        while lastUpdate == self.updated and cycle < 10:
            sleep(0.2)
            cycle = cycle + 1

        if cycle >= 10:
            print('Getting image timed-out.')
            outImage = []
        else:
            outImage = self.image

        ##Testing to send the position out again?
        if self.movableAxis > 0 and MoveOut == True:
            motorX.setAbsPosition(outPos[0])  # at least one motor must be present
            if movableAxis == 2:
                motorY.setAbsPosition(outPos[1])

        return outImage

    def getBeamPosition(self):
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
        gBounds = ([1e4, 0, 1, 10], [1e7, 960, 100, 2e4])

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

        return resX[1], resY[1]

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