import pyjapc
import datetime as dt
import numpy as np
from scipy.optimize import curve_fit
from time import sleep
import matplotlib.pyplot as plt
import pyrbac


bgImages = []
japc = pyjapc.PyJapc(selector = '', incaAcceleratorName = '')
#japc.rbacLogin(loginDialog = True)

def gaussian(x, amp, cen, wid, const):
  return amp*np.exp(-(x-cen)**2/(2*wid**2))+const

class Motor:
  def __init__(self, name, backlash_steps = 0):
    self.name = name
    self.position = -1
    self.lastMotionRel = 0
    self.backlash_steps = backlash_steps
    try:
      if japc.getParam(self.name+'/Acquisition#motor_busy') == False:
        self.position = japc.getParam(self.name+'/Acquisition#position') #This is just to test that acquisition can be made
      else:
        print("Motor is moving!")
    except:
      print('Invalid name!')
  def isBusy(self):
    return japc.getParam(self.name+'/Acquisition#motor_busy')
  
  def getPosition(self): ## This is NEW
    while japc.getParam(self.name+'/Acquisition#motor_busy') == True:
      sleep(0.5)
    self.position = japc.getParam(self.name+'/Acquisition#position_counter') #Uses position in steps! So careful with movements...
    return self.position
   
  def setAbsPosition(self, absPosition): #This is motion in absolute raw steps! No backlash taken into account!!!
    if japc.getParam(self.name+'/Acquisition#motor_busy') == False:
      mv = {'value': absPosition, 'mode': 2, 'units': 2}
      japc.setParam(self.name+'/Move', mv)
    else:
      print('Cannot comply, motor is busy!')
   
  def setRelPosition(self, relPosition): #Motion in relative steps!
    if japc.getParam(self.name+'/Acquisition#motor_busy') == False:  
      
      if relPosition*self.lastMotionRel < 0: #Change of sign
        self.lastMotionRel = relPosition
        relPosition = relPosition + np.sign(relPosition)*self.backlash_steps
      else:
        self.lastMotionRel = relPosition
      
      mv = {'value': relPosition, 'mode': 1, 'units': 2}
      japc.setParam(self.name+'/Move', mv)
      
    else:
      print('Cannot comply, motor is busy!')


class Camera:
  def __init__(self, name, inPos = None, outPos = None, motorX = None, motorY = None): ## Provide motors as Class and positions in steps...
    self.name = name
    self.motorX = motorX
    self.motorY = motorY
    self.inPos = inPos
    self.outPos = outPos
    self.updated = dt.datetime.now()
    self.image = []
    if inPos != None and outPos != None:
      self.movableAxis = len(inPos)
    else:
      self.movableAxis = 0
    self.isIn = -1
    self.subscription = japc.subscribeParam(self.name+'/Acquisition#image1', self.updateImage)
    
  def updateImage(self, parameter, value):
    self.image = value  
    self.updated = dt.datetime.now()
    
  def checkIn(self):
    output = False
    if (self.movableAxis > 0 and self.motorX.getPosition() == self.inPos[0]) and (self.movableAxis == 1 or self.motorY.getPosition() == self.inPos[1]) or self.movableAxis == 0:
      output = True
    return output
  
  def captureImage(self, MoveOut = False):## Should think whether subscription is the way forward or not...
    lastUpdate = self.updated
    if self.checkIn() == False: #Checking if the camera/target is in.
      ##This means that we have to move motor(s)
      self.motorX.setAbsPosition(self.inPos[0]) #at least one motor must be present
      if self.movableAxis == 2:
        self.motorY.setAbsPosition(self.inPos[1])
        while self.motorX.isBusy() or self.motorY.isBusy():
          sleep(0.2)
      while self.motorX.isBusy():
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
    if self.movableAxis>0 and MoveOut == True:
      self.motorX.setAbsPosition(self.outPos[0]) #at least one motor must be present
      if self.movableAxis == 2:
        self.motorY.setAbsPosition(self.outPos[1])
       
    return outImage

  def getBeamInfo(self):
      """
      :returns: resX. resY ([amplitude, center, sigma])
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


def get_positions(plot_range: tuple[int, int], steps: int) -> list:
    """
    Generates a list of positions in a given range with length steps.

    :param plot_range: Minimum and maximum position of the list.
    :param steps: Number of elements in the list.

    :return: A list of positions.
    """
    positions = np.linspace(plot_range[0], plot_range[1], steps)
    return list(positions)
        

def focus(motor: Motor, camera: Camera, plot_range: tuple[int, int], steps: int):
    """
    Focuses a beam.

    :param motor: Motor of the camera to be moved to focus the beam.
    :param camera: Camera to focus the beam on.
    :param plot_range: Range of positions to move the motor in.
    :param steps: Number of steps.

    :return: A picture with horizontal and vertical sigma values depending on the motor position.
    """
    # check is beam exists
    amplitude_threshold = 1001 # maybe set this better
    res_x, res_y = camera.getBeamInfo()
    print("init amplitude: " + str(res_x[0]) + " " + str(res_y[0]))
    if res_x[0] < amplitude_threshold or res_y[0] < amplitude_threshold:
        print("There is currently no beam!")
        return
    
    # # get ref (current position of motor)
    # ref = motor.getPosition()
    #
    # # save original configuration
    # original_ref = ref

    # get positions (x-axis values of the graph)
    positions = get_positions(plot_range, steps)

    # move motor to starting point
    start_pos = positions[0] # minimum of given range
    while motor.isBusy():
        sleep(2)
    motor.setAbsPosition(start_pos)

    # fit gaussian for every position
    sigmas_x = []
    sigmas_y = []
    for i in range(len(positions)):
        image = camera.captureImage()
        
        projection_x = np.sum(image, axis=0)
        g_bounds_x = ([0, 0, 1, 10],[np.inf, len(projection_x), np.inf, np.inf])
        res_x, _ = curve_fit(gaussian, range(len(projection_x)), projection_x, (6e4, np.argmax(projection_x), 10.0, 300.0), bounds = g_bounds_x)
        sigma_x = res_x[2]
        if res_x[0] < amplitude_threshold:
            break
        sigmas_x.append(sigma_x)

        projection_y = np.sum(image, axis=1)
        g_bounds_y = ([0, 0, 1, 10],[np.inf, len(projection_y), np.inf, np.inf])
        res_y, _ = curve_fit(gaussian, range(len(projection_y)), projection_y, (6e4, np.argmax(projection_y), 10.0, 300.0), bounds = g_bounds_y)
        sigma_y = res_y[2]
        sigmas_y.append(sigma_y)
        
        if i == len(positions) - 1:
            print("position " + str(i+1) + " out of " + str(steps))
            break

        while motor.isBusy():
            sleep(2)
        motor.setAbsPosition(positions[i+1])
        print("position " + str(i+1) + " out of " + str(steps))

    positions = list(positions)
    if len(positions) != len(sigmas_x): # if there was premature cut-off
        positions = positions[:len(sigmas_x)]

    min_idx_x = sigmas_x.index(min(sigmas_x))
    while motor.isBusy():
        sleep(2)
    motor.setAbsPosition(positions[min_idx_x]) # go to min sigma x
    graph_focused(positions, sigmas_x, sigmas_y)


def graph_focused(positions, sigmas_x, sigmas_y):
    """
    Graph vertical and horizontal sigma values depending on the motor position.
    """
    if positions == [] or sigmas_x == [] or sigmas_y == []:
        print("There is nothing to graph.")
        return
    
    plt.figure(figsize=(8, 5))

    # Plot curves
    plt.plot(positions, sigmas_x, '-o', label='Sigmas X', color='blue', linewidth=2)
    plt.plot(positions, sigmas_y, '-s', label='Sigmas Y', color='red', linewidth=2)

    # Find minima
    min_idx_x = sigmas_x.index(min(sigmas_x))
    min_idx_y = sigmas_y.index(min(sigmas_y))

    # Mark minima points
    plt.plot(positions[min_idx_x], sigmas_x[min_idx_x], 'bo', markersize=10, label='Min X')
    plt.plot(positions[min_idx_y], sigmas_y[min_idx_y], 'rs', markersize=10, label='Min Y')

    # Annotate positions under points
    plt.annotate(f'{positions[min_idx_x]}', 
                xy=(positions[min_idx_x], sigmas_x[min_idx_x]), 
                xytext=(0, -15), textcoords='offset points', 
                ha='center', color='blue')

    plt.annotate(f'{positions[min_idx_y]}', 
                xy=(positions[min_idx_y], sigmas_y[min_idx_y]), 
                xytext=(0, -15), textcoords='offset points', 
                ha='center', color='red')

    plt.xlabel('Positions')
    plt.ylabel('Sigma Values')
    plt.title('Sigma vs Positions of the Motor')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    client = pyrbac.AuthenticationClient()
    token = client.login_kerberos()
    japc.rbacLogin(token=token)
    
    motor = Motor("TST.cfv-375-bsrdev.mot.3-1")
    camera = Camera("TST.BSRH.CAM.CM")
    japc.startSubscriptions()

    focus(motor, camera, (20000, 60000), 20)

    japc.stopSubscriptions()

if __name__ == "__main__":
    main()
