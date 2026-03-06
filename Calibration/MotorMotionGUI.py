import sys
import time
import threading
import numpy as np
import traceback

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QPushButton, QPlainTextEdit, QMessageBox, QToolBar
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QMetaObject, Q_ARG

import pyjapc
import pyrbac
from accwidgets import rbac

# Ensure we can import modules from the parent directory
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devices_lib import Motor

class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        if text.strip():  # skip empty lines
            self.textWritten.emit(text)

    def flush(self):
        pass  # no-op for compatibility


class MotorMotionGUI(QWidget):
    log_signal = pyqtSignal(str)
    sequence_ended_signal = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Motor Control GUI")
        self.setMinimumSize(400, 500)

        # PyJapc and Rba references
        self.japc = None
        self.japc_token = None

        self.motor_thread = None
        self.motor_running = threading.Event()

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # --- Toolbar with RBAC login ---
        toolbar = QToolBar()
        self.cred_widget = rbac.RbaButton()
        self.cred_widget.loginSucceeded.connect(self.on_login_succeed)
        toolbar.addWidget(self.cred_widget)
        # Attempt an automatic kerberos login if possible
        try:
            self.cred_widget.model.login_with_kerberos(False)
        except Exception as e:
            print(f"Kerberos login failed: {e}")
        main_layout.addWidget(toolbar)

        # --- Inputs ---
        form_layout = QVBoxLayout()

        # Motor Name
        h_name = QHBoxLayout()
        h_name.addWidget(QLabel("Motor Name:"))
        self.motor_name_input = QLineEdit("LHC.BSRH.CMASK-X-TSZ100.B2")
        h_name.addWidget(self.motor_name_input)
        form_layout.addLayout(h_name)

        # Start Position (A)
        h_start = QHBoxLayout()
        h_start.addWidget(QLabel("Start Position A (steps):"))
        self.start_pos_input = QSpinBox()
        self.start_pos_input.setRange(-1000000, 1000000)
        self.start_pos_input.setValue(0)
        h_start.addWidget(self.start_pos_input)
        form_layout.addLayout(h_start)

        # End Position (B)
        h_end = QHBoxLayout()
        h_end.addWidget(QLabel("End Position B (steps):"))
        self.end_pos_input = QSpinBox()
        self.end_pos_input.setRange(-1000000, 1000000)
        self.end_pos_input.setValue(10000)
        h_end.addWidget(self.end_pos_input)
        form_layout.addLayout(h_end)

        # Number of steps
        h_steps = QHBoxLayout()
        h_steps.addWidget(QLabel("Number of Steps:"))
        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 10000)
        self.steps_input.setValue(10)
        h_steps.addWidget(self.steps_input)
        form_layout.addLayout(h_steps)

        # Wait Time
        h_wait = QHBoxLayout()
        h_wait.addWidget(QLabel("Wait Time per Step (s):"))
        self.wait_input = QDoubleSpinBox()
        self.wait_input.setRange(0, 3600)
        self.wait_input.setValue(10.0)
        self.wait_input.setSingleStep(1.0)
        h_wait.addWidget(self.wait_input)
        form_layout.addLayout(h_wait)

        main_layout.addLayout(form_layout)

        # --- Start / Stop Buttons ---
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Sequence")
        self.start_btn.clicked.connect(self.start_sequence)

        self.stop_btn = QPushButton("Stop Sequence")
        self.stop_btn.clicked.connect(self.stop_sequence)
        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        main_layout.addLayout(btn_layout)

        # --- Logs ---
        main_layout.addWidget(QLabel("Logs:"))
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        main_layout.addWidget(self.log_box)

        # Redirect print statements to the log box
        sys.stdout = EmittingStream()
        sys.stdout.textWritten.connect(self.log_message)

        self.log_signal.connect(self._append_log)
        self.sequence_ended_signal.connect(self.on_sequence_ended)

        self.setLayout(main_layout)

    def on_login_succeed(self, token: pyrbac.Token):
        self.japc_token = token
        if self.japc is None:
            self.japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
        try:
            self.japc.rbacLogin(token=self.japc_token)
            self.log_message("RBAC login successful.")
        except Exception as e:
            self.log_message(f"Failed to apply RBAC token to PyJapc: {e}")

    def log_message(self, msg: str):
        self.log_signal.emit(msg)

    def _append_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{ts}] {msg}")

    def start_sequence(self):
        if self.japc is None or self.japc_token is None:
            QMessageBox.warning(self, "Not Logged In", "Please login with RBAC first.")
            return

        motor_name = self.motor_name_input.text().strip()
        start_pos = self.start_pos_input.value()
        end_pos = self.end_pos_input.value()
        num_steps = self.steps_input.value()
        wait_time = self.wait_input.value()

        if not motor_name:
            QMessageBox.warning(self, "Invalid Input", "Motor name cannot be empty.")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.motor_running.set()

        self.motor_thread = threading.Thread(
            target=self.run_sequence,
            args=(motor_name, start_pos, end_pos, num_steps, wait_time),
            daemon=True
        )
        self.motor_thread.start()

    def stop_sequence(self):
        self.log_message("Stopping sequence... (Will stop after current wait/motion)")
        self.motor_running.clear()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def run_sequence(self, motor_name, start_pos, end_pos, num_steps, wait_time):
        self.log_message(f"Starting sequence for motor: {motor_name}")
        self.log_message(f"Path: {start_pos} -> {end_pos} | Steps: {num_steps} | Wait: {wait_time}s")

        try:
            motor = Motor(motor_name, self.japc)
            # Motor initializes busy state based on subscriptions. We should start them.
            self.japc.startSubscriptions()
            time.sleep(1) # Let subscriptions arrive

            if num_steps == 1:
                positions = [start_pos]
            else:
                positions = np.linspace(start_pos, end_pos, num_steps, dtype=int)

            for i, pos in enumerate(positions):
                if not self.motor_running.is_set():
                    self.log_message("Sequence aborted.")
                    break

                self.log_message(f"Step {i+1}/{num_steps}: Moving to {pos}")
                motor.setAbsPosition(pos)

                # Wait for motor to finish moving
                # A timeout logic here prevents infinite hanging if motor gets stuck
                timeout = 60.0
                start_wait = time.time()
                while motor.isBusy():
                    if not self.motor_running.is_set():
                        self.log_message("Abort requested during motion.")
                        break
                    if time.time() - start_wait > timeout:
                        self.log_message(f"Motion timeout ({timeout}s) exceeded while moving to {pos}.")
                        break
                    time.sleep(0.5)

                if not self.motor_running.is_set():
                    break

                # Wait for the specified amount of time at the current position
                self.log_message(f"Arrived at {pos}. Waiting for {wait_time}s...")
                wait_intervals = int(wait_time / 0.5)
                rem_wait = wait_time % 0.5
                for _ in range(wait_intervals):
                    if not self.motor_running.is_set():
                        break
                    time.sleep(0.5)

                if self.motor_running.is_set() and rem_wait > 0:
                    time.sleep(rem_wait)

            if self.motor_running.is_set():
                self.log_message("Sequence completed successfully.")

        except Exception as e:
            self.log_message(f"Error during sequence: {e}")
            traceback.print_exc()
        finally:
            try:
                self.japc.stopSubscriptions()
            except:
                pass
            self.motor_running.clear()

            # Use signal to safely update UI from worker thread
            self.sequence_ended_signal.emit()

    def on_sequence_ended(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MotorMotionGUI()
    gui.show()
    sys.exit(app.exec_())
