# Library for alignment
import torch
import joblib
import numpy as np

# this is just the overall structure

class AlignmentNN:
    def __init__(self, name, feedback_model, scaler_X=None, scaler_y=None):
        self.model = self.load(name, scaler_X, scaler_y)
        self.model.eval()
        self.target = [480, 320] # cam2 center


    def predict(self, X):
        # TODO: add cameras, think about scalability of this class

        input_vector = np.concatenate((cam2_pos, self.target), axis=0).astype(np.float32)
        input_vector_2d = input_vector.reshape(1, -1)
        scaled_input = self.X_scaler.transform(input_vector_2d)
        scaled_tensor = torch.from_numpy(scaled_input).float()

        with torch.no_grad():
            prediction_scaled = self.model(scaled_tensor).numpy()

        prediction_real = self.y_scaler.inverse_transform(prediction_scaled)

        return prediction_real

    def fine_tune(self, X_new, y_new, epochs, lr):
        # load old model weights, keep training
        pass

    def save(self, path):
        pass

    def load(self, file_name, X_scaler_name, y_scaler_name):
        self.model = self.feedbacknn(8, 2)
        self.model.load_state_dict(torch.load(file_name))
        self.X_scaler = joblib.load(X_scaler_name)
        self.y_scaler = joblib.load(y_scaler_name)


class AlignmentPipeline:
    def __init__(self, stage1, stage2, stage3):
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3

    def run_full_alignment(self, initial_state):
        # Stage 1
        steps1 = self.stage1.predict(initial_state)
        # apply steps1 to motors, get new state
        state1 = self._apply_and_measure(steps1)

        # Stage 2
        steps2 = self.stage2.predict(state1)
        state2 = self._apply_and_measure(steps2)

        # Stage 3
        steps3 = self.stage3.predict(state2)
        state3 = self._apply_and_measure(steps3)

        return state3

    def _apply_and_measure(self, steps):
        # send commands to hardware, read new camera/motor state
        pass