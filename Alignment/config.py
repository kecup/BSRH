from devices_lib import Motor, Pneumatics, FWheel
from time import sleep


class CoronagraphDevices:
    def __init__(self, japc):
        self.japc = japc
        self.eaTgt = Pneumatics('LHC.BSRH.IN-OUT.B2', self.japc, 1)
        self.m2Tgt = Motor('LHC.BSRH.TARGET-M3-TS100.B2', self.japc)  # M2 = 42650, HOLE = 22600, OUT = 0 (out can be optimised...)
        self.lsTgt = Pneumatics('LHC.BSRH.IN-OUT.B2', self.japc, 2)
        # TODO: add a block into the CFW in front of the intensifier...
        self.filterND4 = Pneumatics('LHC.BSRH.FILTERS.B2', self.japc, 0)  # This actuates only one filter...
        self.motXCM = Motor('LHC.BSRH.CMASK-X-TSZ100.B2', self.japc) # IN = 60113, OUT = 40333
        self.motYCM = Motor('LHC.BSRH.CMASK-Y-TSZ100.B2', self.japc) # IN = 11940, OUT = 12240
        self.filterCM = FWheel('LHC.BSRH.CM-FWL.B2', self.japc) # 0- ND4, 1- ND3, 2- ND2, 3- ND1, 4 = HOLE


    def allIn(self):  # Not perfect, but this puts EA, LS and m2Tgt into position
        self.lsTgt.moveIn()  ## SAFETY
        self.eaTgt.moveIn()
        self.m2Tgt.setAbsPosition(42650)
        self.motXCM.setAbsPosition(60113)
        self.motYCM.setAbsPosition(11940)
        self.filterCM.setAbsPosition(0)

        #This should make all in... So take bg images

    def measureT1(self): # These are now blocking...
        self.lsTgt.moveIn()  ## SAFETY
        self.eaTgt.moveIn()

    def measureT2(self):
        self.lsTgt.moveIn()  ## SAFETY
        self.eaTgt.moveOut()
        self.m2Tgt.setAbsPosition(42650)
        self.m2Tgt.getPosition() # This is blocking...

    def measureT3(self):
        self.lsTgt.moveIn()  ## SAFETY
        self.eaTgt.moveOut()
        self.m2Tgt.setAbsPosition(22600)
        self.m2Tgt.getPosition()

    def measureT4(self):
        self.lsTgt.moveIn()  ## SAFETY
        self.eaTgt.moveOut()
        self.filterCM.setAbsPosition(0) # Protect camera before removing the T3
        self.m2Tgt.setAbsPosition(0)
        self.motXCM.setAbsPosition(60113)
        self.motYCM.setAbsPosition(11940)
        self.m2Tgt.getPosition()
        self.motXCM.getPosition()
        self.motYCM.getPosition()

    def measureT5(self):
        self.lsTgt.moveIn()  ## SAFETY
        self.eaTgt.moveOut()

        self.motXCM.setAbsPosition(40333)
        self.motYCM.setAbsPosition(12240)
        self.m2Tgt.setAbsPosition(0)

        self.motXCM.getPosition()
        self.motYCM.getPosition()
        self.filterCM.setAbsPosition(4)  # Protect camera before it is out

        self.m2Tgt.getPosition()

    def measureT6(self): # This is tricky, I will get back to this later
        self.eaTgt.moveOut()

        self.motXCM.setAbsPosition(40333)
        self.motYCM.setAbsPosition(12240)
        self.m2Tgt.setAbsPosition(0)

        self.motXCM.getPosition()
        self.motYCM.getPosition()

        self.m2Tgt.getPosition()
        self.filterCM.setAbsPosition(0)
        self.lsTgt.moveOut()  ## SAFETY
        #This puts an ND4 filter at the CM to block also the main camera, one has to then remove the filters accordingly...