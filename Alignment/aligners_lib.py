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
                 X_scaler_name=None, y_scaler_name=None, japc=None, dist_tolerance=None):

        self.dist_tolerance = dist_tolerance
        self.japc = japc
        self.input_size = input_size
        self.output_size = output_size
        self.feedback_nn = feedback_nn
        self.load(model_file_name, X_scaler_name, y_scaler_name)
        self.model.eval()


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



class T3T4AlignmentNN(AlignmentNN):
    """
    Encompasses the logic of alignment for target 3 and target 4. (second optical line)
    """

    def __init__(self, model_file_name=None, input_size=None, output_size=None, feedback_nn=None,
                 X_scaler_name=None, y_scaler_name=None, japc=None, dist_tolerance=15, isr=True):
        super().__init__(model_file_name=model_file_name, input_size=input_size, output_size=output_size,
                         feedback_nn=feedback_nn, X_scaler_name=X_scaler_name, y_scaler_name=y_scaler_name,
                         japc=japc, dist_tolerance=dist_tolerance) # parent init

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
            # TODO

            # LHC Cameras
            # TODO

        # self.cam1_target = [751, 144]  # cam1 target center
        self.cam2_target = [480, 320]  # cam2 target center
        # TODO: check if targets valid also for LHC


    def predict(self, cams_pos):
        """
        Use nn to predict next motor movement.
        """
        # cam1_pos, cam2_pos = cams_pos[0], cams_pos[1]
        cam2_pos = cams_pos[0]

        # input_vector = np.concatenate((cam1_pos, cam2_pos, self.cam1_target, self.cam2_target), axis=0)
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
        cam1_pos = np.array(self.cam1.getBeamPosition())
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

            # motor_prediction = self.predict([cam1_pos, cam2_pos])
            motor_prediction = self.predict([cam2_pos])
            self.move_motors(motor_prediction)

            # cam1_pos = np.array(self.cam1.getBeamPosition())
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
        # sleep(1)


class T5T6AlignmentNN(AlignmentNN): # TODO: adapt for t5 t6, its only copied now
    """
    Encompasses the logic of alignment for target 5 (Lyot stop) and target 6 (main camera). (third/final optical line)
    """

    def __init__(self, model_file_name=None, input_size=None, output_size=None, feedback_nn=None,
                 X_scaler_name=None, y_scaler_name=None, japc=None, dist_tolerance=15, isr=True):
        super().__init__(model_file_name=model_file_name, input_size=input_size, output_size=output_size,
                         feedback_nn=feedback_nn, X_scaler_name=X_scaler_name, y_scaler_name=y_scaler_name,
                         japc=japc, dist_tolerance=dist_tolerance) # parent init

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
            # TODO

            # LHC Cameras
            # TODO

        self.cam1_target = [480, 320] # cam1 target center
        self.cam2_target = [480, 320]  # cam2 target center
        # TODO: check if targets valid (both ISR and LHC)


    def predict(self, cams_pos):
        """
        Use nn to predict next motor movement.
        """
        #TODO: remove cam1? the models trained so far are trained only on cam2 (main cam) - for predict() and align()
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

        self.cam4_abs_pos_in = 15142 # 22.713mm
        self.cam4_abs_pos_out = 40000 # 40mm
        self.cam4 = Motor('TST.cfv-375-bsrdev.mot.4-2', japc, 0) # backlash not measured

        self.nd_filter_abs_pos_out = 0 # 0mm
        self.nd_filter_abs_pos_in = 26000 # 39mm
        self.nd_filter = Motor('TST.cfv-375-bsrdev.mot.4-5', japc, 0) # backlash not measured


    def move_cam4_out(self):

        self.cam4.setAbsPosition(self.cam4_abs_pos_out)
        while self.cam4.isBusy():
            sleep(0.2)

    def move_cam4_in(self):

        self.cam4.setAbsPosition(self.cam4_abs_pos_in)
        while self.cam4.isBusy():
            sleep(0.2)

    def move_nd_filter_in(self):

        self.nd_filter.setAbsPosition(self.nd_filter_abs_pos_in)
        while self.nd_filter.isBusy():
            sleep(0.2)

    def move_nd_filter_out(self):

        self.nd_filter.setAbsPosition(self.nd_filter_abs_pos_out)
        while self.nd_filter.isBusy():
            sleep(0.2)

    def run_full_alignment(self):
        if self.isr:
            # Stage 1: t3 - t4 alignment
            count2 = self.t3_t4_alignment.align()
            print(f"t3 - t4 aligned in {count2} steps")

            self.move_nd_filter_in()
            self.move_cam4_out()

            # Stage 2: t5 - t6 alignment
            count3 = self.t5_t6_alignment.align()
            print(f"t5 - t6 aligned in {count3} steps")

        else:
            # Stage 1: t1 - t2 alignment
            self.t1_t2_alignment.align()

            # Stage 2: t3 - t4 alignment
            self.t3_t4_alignment.align()

            # Stage 3: t5 - t6 alignment
            self.t5_t6_alignment.align()
