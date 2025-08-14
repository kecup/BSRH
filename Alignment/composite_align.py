# Control flow of the full alignment
# There are 3 nns, 2 for ISR
# nn1: t1 + t2 (only LHC)
# nn2: t3 + t4
# nn3: t5 + t6

import pyjapc
import sys
import os
import torch
import joblib
from TrainNN import FeedbackNN1, FeedbackNN2, FeedbackNN3, FeedbackNN4
from aligners_lib import AlignmentPipeline, AlignmentNN

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from devices_lib import Camera, Motor
import pyrbac

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
client = pyrbac.AuthenticationClient()
token = client.login_kerberos()
japc.rbacLogin(token=token)

# ISR Motors
# t3 + t4
motor_x = Motor('TST.cfv-375-bsrdev.mot.3-2', japc, 5)
motor_y = Motor('TST.cfv-375-bsrdev.mot.3-3', japc, 5)
motor_phi = Motor('TST.cfv-375-bsrdev.mot.3-4', japc, 168)
motor_theta = Motor('TST.cfv-375-bsrdev.mot.3-5', japc, 125)

# ISR Cameras
# t3 + t4
cam1 = Camera("TST.BSRH.MON.M2", japc)
cam2 = Camera("TST.BSRH.CAM.CM", japc)


def main():
    align_t1_t2 = AlignmentNN()
    align_t3_t4 = AlignmentNN(name=model_name34, feedback_model=FeedbackNN2, scaler_X=scaler_X_name34, scaler_y=scaler_y_name34)
    align_t5_t6 = AlignmentNN()

    alignment_pipeline = AlignmentPipeline(align_t1_t2, align_t3_t4, align_t5_t6)



if __name__ == "__main__":
    main()