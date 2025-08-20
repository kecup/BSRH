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
from aligners_lib import AlignmentPipeline, T3T4AlignmentNN
from TrainNN import FeedbackNN2

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from devices_lib import Camera, Motor
import pyrbac

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
client = pyrbac.AuthenticationClient()
token = client.login_kerberos()
japc.rbacLogin(token=token)


def main():
    # t1 - t2 Alignment
    # t1_t2_alignment = AlignmentNN()

    # t3 - t4 Alignment
    file_name2 = "Models/isr_t3_t4_v3_only_phi_theta_10770_samples_filtered_model2_cams_targets-steps.pth"
    X_scaler_name2 = "Models/isr_t3_t4_v3_only_phi_theta_10770_samples_filtered_model2_cams_targets-steps_X_scaler.pkl"
    y_scaler_name2 = "Models/isr_t3_t4_v3_only_phi_theta_10770_samples_filtered_model2_cams_targets-steps_y_scaler.pkl"

    t3_t4_alignment = T3T4AlignmentNN(model_file_name=file_name2, input_size=8, output_size=4, feedback_nn=FeedbackNN2,
                 X_scaler_name=X_scaler_name2, y_scaler_name=y_scaler_name2, japc=japc, dist_tolerance=15, isr=True)

    # t5 - t6 Alignment
    # t5_t6_alignment = AlignmentNN()

    alignment_pipeline = AlignmentPipeline(t1_t2_alignment=None, t3_t4_alignment=t3_t4_alignment, t5_t6_alignment=None)
    alignment_pipeline.run_full_alignment()



if __name__ == "__main__":
    main()