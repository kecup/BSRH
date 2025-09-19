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

        # PLUS we need the LS top and EA top, left, right
        self.lsTop = Motor('LHC.BSRH.LYOT-V-UP.B2', self.japc) # We will need to make a creative way of determining how to close it to a proper height
        self.eaTop = Motor('LHC.BSRH.EMASK-V-UP.B2', self.japc)
        self.eaLeft = Motor('LHC.BSRH.EMASK-H-LEFT.B2', self.japc)
        self.eaRight = Motor('LHC.BSRH.EMASK-H-RIGHT.B2', self.japc)

        self.ref_ls = -1
        self.ref_eat = -1
        self.ref_eal = -1
        self.ref_ear = -1


    def allIn(self):  # Not perfect, but this puts EA, LS and m2Tgt into position
        self.lsTgt.moveIn()  ## SAFETY
        self.eaTgt.moveIn()

        # Get references for the apertures
        self.ref_ls = self.lsTop.getPosition()
        self.ref_eat = self.eaTop.getPosition()
        self.ref_eal = self.eaLeft.getPosition()
        self.ref_ear = self.eaRight.getPosition()

        #Open LS and EA
        self.lsTop.setAbsPosition(0)
        self.eaTop.setAbsPosition(0)
        self.eaLeft.setAbsPosition(0)
        self.eaRight.setAbsPosition(0)

        self.m2Tgt.setAbsPosition(42650)
        self.motXCM.setAbsPosition(60113)
        self.motYCM.setAbsPosition(11940)
        self.filterCM.setAbsPosition(0)
        sleep(1)
        #This should make all in... So set correct exposure times, take BG images

    def measureT1(self): # These are now blocking...
        self.lsTgt.moveIn()  ## SAFETY
        self.eaTgt.moveIn()
        sleep(5)

    def measureT2(self):
        self.lsTgt.moveIn()  ## SAFETY
        self.eaTgt.moveOut()
        self.m2Tgt.setAbsPosition(42650)

        sleep(1.5)
        while self.m2Tgt.isBusy():
            sleep(0.5)

    def measureT3(self):
        self.lsTgt.moveIn()  ## SAFETY
        self.eaTgt.moveOut()
        self.m2Tgt.setAbsPosition(22600)

        # sleep(1)
        while self.m2Tgt.isBusy(): #This needs polishing....
            sleep(0.5)

    def measureT4(self):
        self.filterCM.setAbsPosition(0)  # Protect camera before removing the T3
        sleep(1)
        self.lsTgt.moveIn()  ## SAFETY
        self.eaTgt.moveOut()
        self.m2Tgt.setAbsPosition(0)
        self.motXCM.setAbsPosition(60113)
        self.motYCM.setAbsPosition(11940)

        # sleep(1)
        while self.m2Tgt.isBusy() or self.motXCM.isBusy() or self.motYCM.isBusy():
            sleep(0.5)

    def measureT5(self):
        self.lsTgt.moveIn()  ## SAFETY
        self.eaTgt.moveOut()

        self.motXCM.setAbsPosition(36000)
        self.motYCM.setAbsPosition(12240)
        self.m2Tgt.setAbsPosition(0)

        # sleep(1)
        while self.motXCM.isBusy() or self.motYCM.isBusy():
            sleep(0.5)
        self.filterCM.setAbsPosition(4)  # Protect camera before it is out

        sleep(2) # This is the wait for the filter wheel...


    def measureT6(self): # This is tricky, I will get back to this later
        self.eaTgt.moveOut()

        self.motXCM.setAbsPosition(40333)
        self.motYCM.setAbsPosition(12240)
        self.m2Tgt.setAbsPosition(0)

        while self.m2Tgt.isBusy() or self.motXCM.isBusy() or self.motYCM.isBusy():
            sleep(0.5)

        self.filterCM.setAbsPosition(0)
        self.lsTgt.moveOut()  ## SAFETY
        #This puts an ND4 filter at the CM to block also the main camera, one has to then remove the filters accordingly...

    def apertures_to_references(self):
        if self.ref_ls != -1:
            self.lsTop.setAbsPosition(self.ref_ls)
        if self.ref_eat != -1:
            self.eaTop.setAbsPosition(self.ref_eat)
        if self.ref_eal != -1:
            self.eaLeft.setAbsPosition(self.ref_eal)
        if self.ref_ear != -1:
            self.eaRight.setAbsPosition(self.ref_eal)