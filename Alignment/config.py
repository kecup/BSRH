from devices_lib import Camera, Motor, Pneumatics


class CoronagraphDevices:
    def __init__(self, japc):
        self.japc = japc
        self.eaTgt = Pneumatics('LHC.BSRH.IN-OUT.B2', self.japc, 1)
        self.m2Tgt = Motor('LHC.BSRH.TARGET-M3-TS100.B2', self.japc)  # This has 3 positions, M2, HOLE, OUT
        self.lsTgt = Pneumatics('LHC.BSRH.IN-OUT.B2', self.japc, 2)
        # TODO: add a block into the CFW in front of the intensifier...
        self.filterND4 = Pneumatics('LHC.BSRH.FILTERS.B2', self.japc, 0)  # This actuates only one filter...
        self.motXCM = Motor('LHC.BSRH.CMASK-X-TSZ100.B2', self.japc)
        self.motYCM = Motor('LHC.BSRH.CMASK-Y-TSZ100.B2', self.japc)
        self.filterCM = FWheel('LHC.BSRH.CM-FWL.B2', self.japc)


    def allIn(self):  # Not perfect, but this puts EA, LS and m2Tgt into position
        self.lsTgt.moveIn() ## SAFETY
        self.eaTgt.moveIn()
        self.m2Tgt.setAbsPosition()  # TODO: get the correct number in
        self.motXCM.setAbsPosition()
        self.motYCM.setAbsPosition()

