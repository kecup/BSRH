# Library for alignment
import torch
import joblib
import numpy as np
from abc import ABC, abstractmethod
import os
import sys
from time import sleep

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from devices_lib import Camera, Motor

# this is just the overall structure

class AlignmentNN(ABC):
    def __init__(self, model_file_name=None, input_size=None, output_size=None, feedback_nn=None,
                 X_scaler_name=None, y_scaler_name=None, japc=None, dist_tolerance=None, config=None):

        self.dist_tolerance = dist_tolerance
        self.japc = japc
        self.input_size = input_size
        self.output_size = output_size
        self.feedback_nn = feedback_nn
        self.load(model_file_name, X_scaler_name, y_scaler_name)
        self.model.eval()

        self.config = config


    @abstractmethod
    def predict(self, cams_pos):
        pass

    @abstractmethod
    def align(self):
        pass

    @abstractmethod
    def send_to_references(self):
        pass

    @abstractmethod
    def move_motors(self, prediction):
        pass

    def load(self, file_name, X_scaler_name, y_scaler_name):
        self.model = self.feedback_nn(self.input_size, self.output_size)
        self.model.load_state_dict(torch.load(file_name))
        self.X_scaler = joblib.load(X_scaler_name)
        self.y_scaler = joblib.load(y_scaler_name)


class T1T2AlignmentNN(AlignmentNN): # TODO: the whole thing not tested
    """
    Encompasses the logic of alignment for target 1 and target 2. (first optical line)
    """

    def __init__(self, model_file_name=None, input_size=None, output_size=None, feedback_nn=None,
                 X_scaler_name=None, y_scaler_name=None, japc=None, dist_tolerance=15, config=None):
        super().__init__(model_file_name=model_file_name, input_size=input_size, output_size=output_size,
                         feedback_nn=feedback_nn, X_scaler_name=X_scaler_name, y_scaler_name=y_scaler_name,
                         japc=japc, dist_tolerance=dist_tolerance, config=config) # parent init

        self.name = "T1T2Model"

        # LHC Motors
        self.motor_x = Motor('LHC.BSRH.M1-X-TS50.B2', japc, 0)
        self.motor_phi = Motor('LHC.BSRH.STEERING-H.B2', japc, 110)
        self.motor_theta = Motor('LHC.BSRH.STEERING-V.B2', japc, 100)

        #LHC target
        # self.target_ea = Pneumatics('LHC.BSRH.IN-OUT.B2', japc, 1)  # target1

        # LHC Cameras
        self.cam1 = Camera('LHC.BSRH.MON.EA.B2', japc)
        self.cam2 = Camera('LHC.BSRH.MON.TGTM3.B2', japc)

        self.motor_references = {"x":7880, "phi": 219374, "theta": 236211}

        self.cam1_target = [421, 215] # cam1 target center # TODO: y axis has to scale accoringly on both targets..
        self.cam2_target = [310, 205]  # cam2 target center


    def predict(self, cams_pos):
        """
        Use nn to predict next motor movement.
        """

        cam1_pos, cam2_pos = cams_pos[0], cams_pos[1]

        input_vector = np.concatenate((cam1_pos, cam2_pos, self.cam1_target, self.cam2_target), axis=0)
        input_vector_2d = input_vector.reshape(1, -1)
        scaled_input = self.X_scaler.transform(input_vector_2d)
        scaled_tensor = torch.from_numpy(scaled_input).float()

        with torch.no_grad():
            prediction_scaled = self.model(scaled_tensor).numpy()

        prediction_real = self.y_scaler.inverse_transform(prediction_scaled)

        return prediction_real[0]

    def align(self):
        """
        Align to target in a loop until close enough.
        """
        self.config.measureT1()
        cam1_pos = np.array(self.cam1.getBeamPosition())

        self.config.measureT2()
        cam2_pos = np.array(self.cam2.getBeamPosition())
        count = 0

        while (np.abs(cam2_pos[0] - self.cam2_target[0]) > self.dist_tolerance or
               np.abs(cam2_pos[1] - self.cam2_target[1]) > self.dist_tolerance):

            print('Moving, distance to target 2= ', np.sqrt((cam2_pos[0] - self.cam2_target[0]) ** 2 + (cam2_pos[1] - self.cam2_target[1]) ** 2))
            print('Moving, distance to target 1= ', np.sqrt((cam1_pos[0] - self.cam1_target[0]) ** 2 + (cam1_pos[1] - self.cam1_target[1]) ** 2))
            if count > 15:
                print("Not getting closer reasonably fast.")
                break
            if cam2_pos[0] == -1 or cam2_pos[1] == -1:
                print("Beam lost, sending to references.")
                self.send_to_references()
                break

            motor_prediction = self.predict([cam1_pos, cam2_pos])
            self.move_motors(motor_prediction)

            self.config.measureT2() # not necessary
            cam2_pos = np.array(self.cam2.getBeamPosition())

            self.config.measureT1()
            cam1_pos = np.array(self.cam1.getBeamPosition())

            count += 1

        return count

    # def move_target_in(self):
    #     if self.target_ea.getPosition() == 0:  # 0 is OUT and 1 is IN
    #         self.target_ea.changePosition()
    #         sleep(3)
    #
    # def move_target_out(self):
    #     if self.target_ea.getPosition() == 1:  # 0 is OUT and 1 is IN
    #         self.target_ea.changePosition()
    #         sleep(3)

    def send_to_references(self):
        """
        Moves motors to pre-measured references.
        """
        self.motor_x.setAbsPosition(self.motor_references["x"])
        self.motor_phi.setAbsPosition(self.motor_references["phi"])
        self.motor_theta.setAbsPosition(self.motor_references["theta"])
        while self.motor_x.isBusy() or self.motor_phi.isBusy() or self.motor_theta.isBusy():
            sleep(0.2)

    def move_motors(self, prediction):
        """
        Move motors by predicted steps.
        """
        self.motor_x.setAbsPosition(prediction[0])
        self.motor_phi.setRelPosition(prediction[1])
        self.motor_theta.setRelPosition(prediction[2])
        while self.motor_x.isBusy() or self.motor_phi.isBusy() or self.motor_theta.isBusy():
            sleep(0.2)


class T3T4AlignmentNN(AlignmentNN):
    """
    Encompasses the logic of alignment for target 3 and target 4. (second optical line)
    """

    def __init__(self, model_file_name=None, input_size=None, output_size=None, feedback_nn=None,
                 X_scaler_name=None, y_scaler_name=None, japc=None, dist_tolerance=15, config=None, isr=True):
        super().__init__(model_file_name=model_file_name, input_size=input_size, output_size=output_size,
                         feedback_nn=feedback_nn, X_scaler_name=X_scaler_name, y_scaler_name=y_scaler_name,
                         japc=japc, dist_tolerance=dist_tolerance, config=config) # parent init

        self.name = "T3T4Model"
        self.isr = isr # whether aligning on ISR or LHC

        if self.isr:
            # ISR Motors
            self.motor_x = Motor('TST.cfv-375-bsrdev.mot.3-2', self.japc, 5) # not used
            self.motor_y = Motor('TST.cfv-375-bsrdev.mot.3-3', self.japc, 5) # not used
            self.motor_phi = Motor('TST.cfv-375-bsrdev.mot.3-4', self.japc, 168)
            self.motor_theta = Motor('TST.cfv-375-bsrdev.mot.3-5', self.japc, 125)

            self.motor_references = {"x": 40385, "y": 41740, "phi": 140381, "theta": 163301}

            # ISR Cameras
            self.cam1 = Camera("TST.BSRH.MON.M2", self.japc)
            self.cam2 = Camera("TST.BSRH.CAM.CM", self.japc)
        else:
            pass
            # LHC Motors
            self.motor_x = Motor('LHC.BSRH.M3-X-TSZ100.B2', self.japc, 5)  # not used
            self.motor_y = Motor('LHC.BSRH.M3-Y-TSZ100.B2', self.japc, 5)  # not used
            self.motor_phi = Motor('LHC.BSRH.M3-PHI.B2', self.japc, 65)
            self.motor_theta = Motor('LHC.BSRH.M3-THETA.B2', self.japc, 50)

            self.motor_references = {"x": 31333, "y": 29373, "phi": 82285, "theta": 70097}

            # LHC Cameras
            self.cam1 = Camera("LHC.BSRH.MON.TGTM3.B2", self.japc, True)
            self.cam2 = Camera("LHC.BSRH.CAM.CM.B2", self.japc, True)

        # self.cam1_target = [751, 144]  # cam1 target center
        self.cam2_target = [698, 94]  # cam2 target center


    def predict(self, cams_pos):
        """
        Use nn to predict next motor movement.
        """
        cam2_pos = cams_pos[0]

        input_vector = np.concatenate((cam2_pos, self.cam2_target), axis=0)
        input_vector_2d = input_vector.reshape(1, -1)
        scaled_input = self.X_scaler.transform(input_vector_2d)
        scaled_tensor = torch.from_numpy(scaled_input).float()

        with torch.no_grad():
            prediction_scaled = self.model(scaled_tensor).numpy()

        prediction_real = self.y_scaler.inverse_transform(prediction_scaled)

        return prediction_real[0]

    def align(self):
        """
        Align to target in a loop until close enough.
        """
        self.config.measureT4()
        sleep(2)
        cam2_pos = np.array(self.cam2.getBeamPosition())
        count = 0

        while (np.abs(cam2_pos[0] - self.cam2_target[0]) > self.dist_tolerance or
               np.abs(cam2_pos[1] - self.cam2_target[1]) > self.dist_tolerance):
            if count > 15:
                print("Not getting closer reasonably fast.")
                break
            if cam2_pos[0] == -1 or cam2_pos[1] == -1:
                print("Beam lost, sending to references.")
                self.send_to_references()
                break

            motor_prediction = self.predict([cam2_pos])
            self.move_motors(motor_prediction)
            cam2_pos = np.array(self.cam2.getBeamPosition())
            count += 1
        return count


    def send_to_references(self):
        """
        Moves motors to pre-measured references.
        """
        self.motor_x.setAbsPosition(self.motor_references["x"])
        self.motor_y.setAbsPosition(self.motor_references["y"])
        self.motor_phi.setAbsPosition(self.motor_references["phi"])
        self.motor_theta.setAbsPosition(self.motor_references["theta"])
        while self.motor_x.isBusy() or self.motor_y.isBusy() or self.motor_phi.isBusy() or self.motor_theta.isBusy():
            sleep(0.2)

    def move_motors(self, prediction):
        """
        Move motors by predicted steps - only phi and theta motors move.
        """
        self.motor_phi.setRelPosition(prediction[0])
        self.motor_theta.setRelPosition(prediction[1])
        while self.motor_phi.isBusy() or self.motor_theta.isBusy():
            sleep(0.2)
        sleep(1)


class T5T6AlignmentNN(AlignmentNN): # TODO: adapt for t5 t6, its only copied now
    """
    Encompasses the logic of alignment for target 5 (Lyot stop) and target 6 (main camera). (third/final optical line)
    """

    def __init__(self, model_file_name=None, input_size=None, output_size=None, feedback_nn=None,
                 X_scaler_name=None, y_scaler_name=None, japc=None, dist_tolerance=15, config=None, isr=True):
        super().__init__(model_file_name=model_file_name, input_size=input_size, output_size=output_size,
                         feedback_nn=feedback_nn, X_scaler_name=X_scaler_name, y_scaler_name=y_scaler_name,
                         japc=japc, dist_tolerance=dist_tolerance, config=config) # parent init

        self.name = "T5T6Model"
        self.isr = isr # whether aligning on ISR or LHC

        if self.isr:
            # ISR Motors
            self.motor_phi = Motor('TST.cfv-375-bsrdev.mot.5-1', japc, 11)
            self.motor_theta = Motor('TST.cfv-375-bsrdev.mot.5-2', japc, 8)

            # ISR Cameras
            self.cam1 = Camera("TST.BSRH.MON.LS", japc)  # Lyot stop
            self.cam2 = Camera("TST.BSRH.CAM.MAIN", japc)  # Main camera

            self.motor_references = {"phi": 13444, "theta": 13360}
        else:
            pass
            # LHC Motors
            self.motor_phi = Motor('LHC.BSRH.M3-PHI.B2', self.japc, 64)
            self.motor_theta = Motor('LHC.BSRH.M3-THETA.B2', self.japc, 106)
            #self.motor_phi = Motor('LHC.BSRH.FOLDING-H.B2', japc, 45)
             #self.motor_theta = Motor('LHC.BSRH.FOLDING-V.B2', japc, 0)

            # LHC Cameras
            self.cam2 = Camera('LHC.BSRH.BTVDC-CAM.MAIN', self.japc, False)

            self.motor_references = {"phi": 82285, "theta": 70097}
            self.occulterPosition = [484, 300, 45] # This is to call the corect get beam positionin infinite alignment...

        self.cam2_target = [484, 304]  # cam2 target center
        # TODO: check if targets valid (both ISR and LHC)


    def old_predict(self, cams_pos, target = None):
        """
        Use nn to predict next motor movement.
        """
        #TODO: the models trained so far are trained only on cam2 (main cam) - for predict() and align()
        # - it's gonna be a problem if the model is different

        cam2_pos = cams_pos[0]

        if target == None:
            input_vector = np.concatenate((cam2_pos, self.cam2_target), axis=0)
        else:
            input_vector = np.concatenate((cam2_pos, target), axis=0)
        input_vector_2d = input_vector.reshape(1, -1)
        scaled_input = self.X_scaler.transform(input_vector_2d)
        scaled_tensor = torch.from_numpy(scaled_input).float()

        with torch.no_grad():
            prediction_scaled = self.model(scaled_tensor).numpy()

        prediction_real = self.y_scaler.inverse_transform(prediction_scaled)

        return prediction_real[0]


    def predict(self, cams_pos, target = None):
        """
        Use nn to predict next motor movement.
        """
        #TODO: the models trained so far are trained only on cam2 (main cam) - for predict() and align()
        # - it's gonna be a problem if the model is different

        cam2_pos = cams_pos[0]

        if target == None:
            input_vector = np.concatenate((cam2_pos, self.cam2_target), axis=0)
        else:
            input_vector = np.concatenate((cam2_pos, target), axis=0)

        distX = input_vector[2]-input_vector[0]
        distY = input_vector[3]-input_vector[1]

        if distX < 10:
            stepsX = distX*1.0
        else:
            stepsX = distX*1.8

        if distY < 10:
            stepsY = distY*-0.9
        else:
            stepsY = distY*-1.4

        return [stepsX, stepsY]

    def align(self, target = None, bg = None): # Just in case we would want to change the target...
        """
        Align to target in a loop until close enough.
        """
        if bg != None:
            self.cam2.setBg(bg)

        self.config.measureT6()
        sleep(2)
        count = 0

        # TODO: Think hard about how this should work!!!
        cam2_pos = [-1, -1]
        while cam2_pos[0] == -1 or cam2_pos[1] == -1 and count < 5:
            cam2_pos = np.array(self.cam2.getBeamPosition()) # We check to make sure that this is not one of a kind issue, we should include BG though... that is in the main thread...
            count += 1
        if count == 5:
            cam2_pos = np.array(self.cam2.getBeamPosition(com = True))

        count = 0

        while (np.abs(cam2_pos[0] - self.cam2_target[0]) > self.dist_tolerance or np.abs(cam2_pos[1] - self.cam2_target[1]) > self.dist_tolerance): # This is not what I asked for...
            if count > 15:
                print("Not getting closer reasonably fast.")
                break
            if cam2_pos[0] == -1 or cam2_pos[1] == -1:
                print("Beam lost, sending to references.") # This should be last resort...
                self.send_to_references()
                sleep(1)
                #break

            motor_prediction = self.predict([cam2_pos], target)
            self.move_motors(motor_prediction)

            cam2_pos = np.array(self.cam2.getBeamPosition())
            count += 1

        return count

    def send_to_references(self):
        """
        Moves motors to pre-measured references.
        """
        self.motor_phi.setAbsPosition(self.motor_references["phi"])
        self.motor_theta.setAbsPosition(self.motor_references["theta"])
        while self.motor_phi.isBusy() or self.motor_theta.isBusy():
            sleep(0.2)

    def move_motors(self, prediction):
        """
        Move motors by predicted steps.
        """
        self.motor_phi.setRelPosition(prediction[0])
        self.motor_theta.setRelPosition(prediction[1])
        while self.motor_phi.isBusy() or self.motor_theta.isBusy():
            sleep(0.2)
        sleep(1) # Don't forget to sleep for some time to make sure that we get correct measurements!


class AlignmentPipeline:
    def __init__(self, t1_t2_alignment, t3_t4_alignment, t5_t6_alignment, japc, isr=True):

        self.isr = isr

        if self.isr:
            self.t3_t4_alignment = t3_t4_alignment
            self.t5_t6_alignment = t5_t6_alignment

        else:
            self.t1_t2_alignment = t1_t2_alignment
            self.t3_t4_alignment = t3_t4_alignment
            self.t5_t6_alignment = t5_t6_alignment



        # self.cam4_abs_pos_in = 15142 # 22.713mm
        # self.cam4_abs_pos_out = 40000 # 40mm
        # self.cam4 = Motor('TST.cfv-375-bsrdev.mot.4-2', japc, 0) # backlash not measured
        #
        # self.nd_filter_abs_pos_out = 0 # 0mm
        # self.nd_filter_abs_pos_in = 26000 # 39mm
        # self.nd_filter = Motor('TST.cfv-375-bsrdev.mot.4-5', japc, 0) # backlash not measured


    # def move_cam4_out(self):
    #
    #     self.cam4.setAbsPosition(self.cam4_abs_pos_out)
    #     while self.cam4.isBusy():
    #         sleep(0.2)
    #
    # def move_cam4_in(self):
    #
    #     self.cam4.setAbsPosition(self.cam4_abs_pos_in)
    #     while self.cam4.isBusy():
    #         sleep(0.2)
    #
    # def move_nd_filter_in(self):
    #
    #     self.nd_filter.setAbsPosition(self.nd_filter_abs_pos_in)
    #     while self.nd_filter.isBusy():
    #         sleep(0.2)
    #
    # def move_nd_filter_out(self):
    #
    #     self.nd_filter.setAbsPosition(self.nd_filter_abs_pos_out)
    #     while self.nd_filter.isBusy():
    #         sleep(0.2)

    def run_full_alignment(self):
        if self.isr:
            # Stage 1: t3 - t4 alignment
            count2 = self.t3_t4_alignment.align()
            print(f"t3 - t4 aligned in {count2} steps")

            # self.move_nd_filter_in()
            # self.move_cam4_out()

            # Stage 2: t5 - t6 alignment
            count3 = self.t5_t6_alignment.align()
            print(f"t5 - t6 aligned in {count3} steps")

        else:
            # Stage 1: t1 - t2 alignment
            count1 = self.t1_t2_alignment.align()
            print(f"t1 - t2 aligned in {count1} steps")

            # # Stage 2: t3 - t4 alignment
            # count2 = self.t3_t4_alignment.align()
            # print(f"t3 - t4 aligned in {count2} steps")
            #
            # # self.move_nd_filter_in()
            # # self.move_cam4_out()
            #
            # # Stage 3: t5 - t6 alignment
            # count3 = self.t5_t6_alignment.align()
            # print(f"t5 - t6 aligned in {count3} steps")

