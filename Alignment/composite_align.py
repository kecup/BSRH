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
from TrainNN import FeedbackNN1, FeedbackNN2, FeedbackNN3, FeedbackNN4, FeedbackNN5, FeedbackNN6, FeedbackNN7
from aligners_lib import AlignmentPipeline, T1T2AlignmentNN, T3T4AlignmentNN, T5T6AlignmentNN
from time import sleep


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import pyrbac
from config import CoronagraphDevices

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
client = pyrbac.AuthenticationClient()
token = client.login_kerberos()
japc.rbacLogin(token=token)


def main():

    lhc_config = CoronagraphDevices(japc)
    # isr_config = ...

    # t1 - t2 Alignment
    file_name1 = "Models/lhc_t1_t2_v5_model1_cams_targets-steps.pth"
    X_scaler_name1 = "Models/lhc_t1_t2_v5_model1_cams_targets-steps_X_scaler.pkl"
    y_scaler_name1 = "Models/lhc_t1_t2_v5_model1_cams_targets-steps_y_scaler.pkl"

    # only cam2 model
    t1_t2_alignment = T1T2AlignmentNN(model_file_name=file_name1, input_size=8, output_size=3, feedback_nn=FeedbackNN7,
                                      X_scaler_name=X_scaler_name1, y_scaler_name=y_scaler_name1, japc=japc,
                                      dist_tolerance=10, config=lhc_config)

    # t3 - t4 Alignment
    file_name2 = "Models/isr_t3_t4_v4_filtered_model3_only_cam2_target2-steps.pth"
    X_scaler_name2 = "Models/isr_t3_t4_v4_filtered_model3_only_cam2_target2-steps_X_scaler.pkl"
    y_scaler_name2 = "Models/isr_t3_t4_v4_filtered_model3_only_cam2_target2-steps_y_scaler.pkl"

    # only cam2 model
    t3_t4_alignment = T3T4AlignmentNN(model_file_name=file_name2, input_size=4, output_size=2, feedback_nn=FeedbackNN6,
                 X_scaler_name=X_scaler_name2, y_scaler_name=y_scaler_name2, japc=japc, dist_tolerance=15, config=lhc_config, isr=False)

    # t5 - t6 Alignment
    file_name3 = "Models/isr_t5_t6_v3_model1_cam2_target2-steps.pth"
    X_scaler_name3 = "Models/isr_t5_t6_v3_model1_cam2_target2-steps_X_scaler.pkl"
    y_scaler_name3 = "Models/isr_t5_t6_v3_model1_cam2_target2-steps_y_scaler.pkl"

    # only cam2 model now
    t5_t6_alignment = T5T6AlignmentNN(model_file_name=file_name3, input_size=4, output_size=2, feedback_nn=FeedbackNN6,
                                      X_scaler_name=X_scaler_name3, y_scaler_name=y_scaler_name3, japc=japc,
                                      dist_tolerance=15, config=lhc_config, isr=False)


    alignment_pipeline = AlignmentPipeline(t1_t2_alignment=t1_t2_alignment, t3_t4_alignment=t3_t4_alignment,
                                           t5_t6_alignment=t5_t6_alignment, japc=japc, isr=False)


    ######## main alignment ########
    japc.startSubscriptions()
    sleep(2)

    lhc_config.allIn()
    alignment_pipeline.run_full_alignment()

    japc.stopSubscriptions()

if __name__ == "__main__":
    main()