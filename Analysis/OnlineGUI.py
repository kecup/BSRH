import sys
import os
import threading
import queue
import time
import argparse
import numpy as np
import dill as pickle
import zstandard as zstd
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QPushButton, QFileDialog)
from PyQt5.QtCore import QTimer, Qt, QObject, pyqtSignal
import pyqtgraph as pg

try:
    from accwidgets import rbac
    import pyrbac
except ImportError:
    rbac = None
    pyrbac = None

from scipy.optimize import curve_fit

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

class OfflineDataReplayer(threading.Thread):
    def __init__(self, directory, data_queue, gui_ref):
        super().__init__()
        self.directory = directory
        self.data_queue = data_queue
        self.gui_ref = gui_ref
        self.running = True
        self.daemon = True
        self.dctx = zstd.ZstdDecompressor()

    def run(self):
        files = sorted([f for f in os.listdir(self.directory) if f.endswith('.pkl.zst')])
        print(f"Found {len(files)} files in {self.directory}")
        for file in files:
            if not self.running:
                break
            full_path = os.path.join(self.directory, file)
            print(f"Processing {file}...")
            try:
                with open(full_path, 'rb') as f:
                    with self.dctx.stream_reader(f) as reader:
                        data_chunk = pickle.load(reader)

                for event in data_chunk:
                    if not self.running:
                        break

                    try:
                        image2D = event.cam_data['image2D']
                        device_data = event.device_data

                        delay_slots = device_data['GATE-INT']['delaySlots']
                        num_pulses = device_data['GATE-INT']['numberOfPulses']
                        bunch_index = int(delay_slots)

                        bunches = device_data['FBCT1']['averageBunchIntensities']
                        if 0 <= bunch_index < len(bunches):
                            fbct_charge = bunches[bunch_index]
                        else:
                            fbct_charge = 0.0

                        event_data = {
                            'image2D': image2D,
                            'bunch_index': bunch_index,
                            'fbct_charge': fbct_charge,
                            'num_pulses': num_pulses,
                            'timestamp': event.timestamp
                        }

                        # Block if queue is full to not overwhelm memory
                        self.data_queue.put(event_data, block=True, timeout=5.0)

                        # Small sleep to yield to GUI (simulating arrival)
                        time.sleep(0.05)
                    except queue.Full:
                        pass
                    except Exception as e:
                        print(f"Error parsing event from file: {e}")
            except Exception as e:
                print(f"Error loading file {file}: {e}")

        print("Offline replay finished.")

    def stop(self):
        self.running = False


class DataAcquisition(QObject):
    def __init__(self, data_queue):
        super().__init__()
        self.data_queue = data_queue

        try:
            import pyjapc
            self.japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
        except ImportError:
            print("pyjapc not available, running in mock mode.")
            self.japc = None

        self.latest_fbct = None
        self.latest_gate_settings = None

        self.camera_device = 'LHC.BSRH.BTVDC-CAM.MAIN/LastImage'
        self.gate_device = 'LHC.BSRH.GATE-GEN.5L4.B2/Setting1'
        self.fbct_device = 'LHC.BCTFR.A6R4.B2/Acquisition'

    def set_token(self, token):
        if self.japc and pyrbac:
            try:
                self.japc.rbacLogin(token=token)
                self.start_subscriptions()
            except Exception as e:
                print(f"RBAC Login failed: {e}")

    def start_subscriptions(self):
        if not self.japc:
            return

        self.japc.subscribeParam(self.fbct_device, self.fbct_callback)
        self.japc.subscribeParam(self.gate_device, self.gate_callback)
        self.japc.subscribeParam(self.camera_device, self.camera_callback, getHeader=True)

        try:
            self.japc.startSubscriptions()
        except Exception as e:
            print(f"Failed to start subscriptions: {e}")

    def stop_subscriptions(self):
        if self.japc:
            self.japc.stopSubscriptions()

    def fbct_callback(self, name, value):
        self.latest_fbct = value

    def gate_callback(self, name, value):
        self.latest_gate_settings = value

    def camera_callback(self, name, value, header):
        if self.latest_fbct is None or self.latest_gate_settings is None:
            return

        try:
            if isinstance(self.latest_fbct, dict):
                bunches = self.latest_fbct.get('averageBunchIntensities', [])
            else:
                bunches = self.latest_fbct['averageBunchIntensities']

            if isinstance(self.latest_gate_settings, dict):
                delay_slots = self.latest_gate_settings.get('delaySlots', 0)
                num_pulses = self.latest_gate_settings.get('numberOfPulses', 1)
            else:
                delay_slots = self.latest_gate_settings['delaySlots']
                num_pulses = self.latest_gate_settings['numberOfPulses']

            image2D = value['image2D']

        except Exception as e:
            print(f"Error parsing data in callback: {e}")
            return

        bunch_index = int(delay_slots)
        if 0 <= bunch_index < len(bunches):
            fbct_charge = bunches[bunch_index]
        else:
            fbct_charge = 0.0

        event_data = {
            'image2D': image2D,
            'bunch_index': bunch_index,
            'fbct_charge': fbct_charge,
            'num_pulses': num_pulses,
            'timestamp': time.time()
        }

        try:
            self.data_queue.put_nowait(event_data)
        except queue.Full:
            pass

class AnalysisWorker(threading.Thread):
    def __init__(self, data_queue, gui_ref):
        super().__init__()
        self.data_queue = data_queue
        self.gui_ref = gui_ref
        self.running = True
        self.daemon = True

        self.background_image = None
        self.intensifier_gain = 5651.0

        self.x1, self.x2 = 234, 684
        self.y1, self.y2 = 154, 454

    def run(self):
        while self.running:
            try:
                event = self.data_queue.get(timeout=0.1)
                self.process_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in analysis thread: {e}")

    def process_event(self, event):
        image = event['image2D']
        fbct_charge = event['fbct_charge']
        num_pulses = event['num_pulses']
        bunch_index = event['bunch_index']

        if fbct_charge <= 0:
            self.background_image = image.copy()
            return

        if self.background_image is None:
            return

        maskX = self.gui_ref.spin_maskX.value()
        maskY = self.gui_ref.spin_maskY.value()
        maskR = self.gui_ref.spin_maskR.value()
        mask_factor = self.gui_ref.spin_mask_factor.value()
        sigma_val = self.gui_ref.spin_sigma.value()

        try:
            bg_subtracted = image - self.background_image
        except ValueError: # Shapes might not match perfectly in some weird cases
            return

        y_indices, x_indices = np.ogrid[:bg_subtracted.shape[0], :bg_subtracted.shape[1]]
        mask_area = (x_indices - maskX)**2 + (y_indices - maskY)**2 <= maskR**2
        bg_subtracted[mask_area] = bg_subtracted[mask_area] * mask_factor

        img_sum = np.sum(bg_subtracted)
        if img_sum > 0 and num_pulses > 0:
            cPc = (fbct_charge * num_pulses) / (img_sum * self.intensifier_gain)
        else:
            cPc = 0.0

        if cPc == 0.0:
            return

        self.gui_ref.latest_image = bg_subtracted

        try:
            projection_x = np.sum(bg_subtracted[self.y1:self.y2, :], axis=0)
            projection_y = np.sum(bg_subtracted[:, self.x1:self.x2], axis=1)
        except Exception as e:
            print(f"Error cropping: {e}")
            return

        x_temp = []
        x_temp.extend(range(int(max(0, maskX - maskR))))
        x_temp.extend(range(int(min(len(projection_x), maskX + maskR)), len(projection_x)))

        resX_center = maskX
        if len(x_temp) > 0:
            try:
                resX, _ = curve_fit(gaussian, x_temp, projection_x[x_temp], (45000, maskX, 35.0))
                if not (maskX - maskR < resX[1] < maskX + maskR):
                    resX_center = resX[1]
            except:
                pass

        y_temp = []
        y_temp.extend(range(int(max(0, maskY - maskR))))
        y_temp.extend(range(int(min(len(projection_y), maskY + maskR)), len(projection_y)))

        resY_center = maskY
        if len(y_temp) > 0:
            try:
                resY, _ = curve_fit(gaussian, y_temp, projection_y[y_temp], (45000, maskY, 15.0))
                if not (maskY - maskR < resY[1] < maskY + maskR):
                    resY_center = resY[1]
            except:
                pass

        pixelSizeH = 19.6
        pixelSizeV = 20.3
        SIGMA_TO_MM_X = 263.0
        SIGMA_TO_MM_Y = 361.0

        xl1 = int(resX_center - sigma_val * SIGMA_TO_MM_X / pixelSizeH)
        xr1 = int(resX_center + sigma_val * SIGMA_TO_MM_X / pixelSizeH)

        chargeX = 0
        if xl1 > 0:
            chargeX += np.sum(projection_x[:xl1])
        if xr1 < len(projection_x):
            chargeX += np.sum(projection_x[xr1:])

        haloX = chargeX * cPc * self.intensifier_gain / num_pulses

        yl1 = int(resY_center - sigma_val * SIGMA_TO_MM_Y / pixelSizeV)
        yr1 = int(resY_center + sigma_val * SIGMA_TO_MM_Y / pixelSizeV)

        chargeY = 0
        if yl1 > 0:
            chargeY += np.sum(projection_y[:yl1])
        if yr1 < len(projection_y):
            chargeY += np.sum(projection_y[yr1:])

        haloY = chargeY * cPc * self.intensifier_gain / num_pulses

        if 0 <= bunch_index < len(self.gui_ref.halo_h_data):
            self.gui_ref.halo_h_data[bunch_index] = haloX
            self.gui_ref.halo_v_data[bunch_index] = haloY

    def stop(self):
        self.running = False


class OnlineAnalysisGUI(QWidget):
    def __init__(self, offline_dir=None):
        super().__init__()
        self.setWindowTitle("Online Halo Analysis GUI")
        self.resize(1000, 800)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.setup_config_panel()

        self.image_view = pg.ImageView()
        self.layout.addWidget(self.image_view, stretch=2)

        self.plot_v = pg.PlotWidget(title="Vertical Halo")
        self.plot_v.setLabel('bottom', 'Bunch Slot')
        self.plot_v.setLabel('left', 'Halo Charge')
        self.curve_v = self.plot_v.plot(pen=None, symbol='o')
        self.label_sum_v = pg.TextItem(text="Total Vertical Halo: 0", anchor=(1, 0), color=(200, 200, 200))
        self.plot_v.addItem(self.label_sum_v)
        self.layout.addWidget(self.plot_v, stretch=1)

        self.plot_h = pg.PlotWidget(title="Horizontal Halo")
        self.plot_h.setLabel('bottom', 'Bunch Slot')
        self.plot_h.setLabel('left', 'Halo Charge')
        self.curve_h = self.plot_h.plot(pen=None, symbol='o')
        self.label_sum_h = pg.TextItem(text="Total Horizontal Halo: 0", anchor=(1, 0), color=(200, 200, 200))
        self.plot_h.addItem(self.label_sum_h)
        self.layout.addWidget(self.plot_h, stretch=1)

        self.halo_h_data = np.zeros(3564)
        self.halo_v_data = np.zeros(3564)

        self.plot_v.sigRangeChanged.connect(self.update_label_positions)
        self.plot_h.sigRangeChanged.connect(self.update_label_positions)

        self.data_queue = queue.Queue(maxsize=5)
        self.latest_image = None

        self.worker = AnalysisWorker(self.data_queue, self)
        self.worker.start()

        self.acq = None
        self.replayer = None

        if offline_dir:
            self.start_offline_replay(offline_dir)
        else:
            self.acq = DataAcquisition(self.data_queue)
            # Subscriptions will start upon successful RBAC login

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_gui)
        self.update_timer.start(1000)

    def setup_config_panel(self):
        config_group = QGroupBox("Configuration")
        config_layout = QHBoxLayout()

        config_layout.addWidget(QLabel("maskX:"))
        self.spin_maskX = QSpinBox()
        self.spin_maskX.setRange(0, 2000)
        self.spin_maskX.setValue(474)
        config_layout.addWidget(self.spin_maskX)

        config_layout.addWidget(QLabel("maskY:"))
        self.spin_maskY = QSpinBox()
        self.spin_maskY.setRange(0, 2000)
        self.spin_maskY.setValue(307)
        config_layout.addWidget(self.spin_maskY)

        config_layout.addWidget(QLabel("maskR:"))
        self.spin_maskR = QSpinBox()
        self.spin_maskR.setRange(0, 1000)
        self.spin_maskR.setValue(55)
        config_layout.addWidget(self.spin_maskR)

        config_layout.addWidget(QLabel("mask factor:"))
        self.spin_mask_factor = QDoubleSpinBox()
        self.spin_mask_factor.setRange(0.1, 100.0)
        self.spin_mask_factor.setValue(8.3)
        self.spin_mask_factor.setSingleStep(0.1)
        config_layout.addWidget(self.spin_mask_factor)

        config_layout.addWidget(QLabel("sigma:"))
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 10.0)
        self.spin_sigma.setValue(3.5)
        self.spin_sigma.setSingleStep(0.1)
        config_layout.addWidget(self.spin_sigma)

        self.btn_live = QPushButton("Start Live Data")
        self.btn_live.setCheckable(True)
        self.btn_live.clicked.connect(self.toggle_live_data)
        config_layout.addWidget(self.btn_live)

        if rbac:
            self.cred_widget = rbac.RbaButton()
            self.cred_widget.loginSucceeded.connect(self.on_login_succeed)
            config_layout.addWidget(self.cred_widget)
            try:
                self.cred_widget.model.login_with_kerberos(False)
            except:
                pass

        self.btn_offline = QPushButton("Run Offline Data")
        self.btn_offline.clicked.connect(self.open_offline_dialog)
        config_layout.addWidget(self.btn_offline)

        config_group.setLayout(config_layout)
        self.layout.addWidget(config_group)

    def on_login_succeed(self, token):
        if self.acq and hasattr(self, 'btn_live') and self.btn_live.isChecked():
            self.acq.set_token(token)

    def toggle_live_data(self, checked):
        if checked:
            self.btn_live.setText("Stop Live Data")
            if self.acq:
                # If we have a token, start. If not, it should wait for RBAC.
                # Here we blindly attempt, the acq layer should handle it safely.
                if rbac and hasattr(self, 'cred_widget') and self.cred_widget.model.token:
                    self.acq.set_token(self.cred_widget.model.token)
                else:
                    self.acq.start_subscriptions()
        else:
            self.btn_live.setText("Start Live Data")
            if self.acq:
                self.acq.stop_subscriptions()

    def open_offline_dialog(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Offline Data Directory")
        if dir_path:
            self.start_offline_replay(dir_path)

    def start_offline_replay(self, dir_path):
        if self.acq:
            self.acq.stop_subscriptions()
            self.acq = None

        if self.replayer:
            self.replayer.stop()
            self.replayer.join()

        # Reset data for new run
        self.halo_h_data = np.zeros(3564)
        self.halo_v_data = np.zeros(3564)
        self.latest_image = None
        self.worker.background_image = None

        print(f"Starting offline replay from {dir_path}")
        self.replayer = OfflineDataReplayer(dir_path, self.data_queue, self)
        self.replayer.start()

    def update_label_positions(self):
        view_rect_v = self.plot_v.viewRect()
        self.label_sum_v.setPos(view_rect_v.right(), view_rect_v.top())

        view_rect_h = self.plot_h.viewRect()
        self.label_sum_h.setPos(view_rect_h.right(), view_rect_h.top())

    def update_gui(self):
        if getattr(self, 'latest_image', None) is not None:
            self.image_view.setImage(self.latest_image, autoRange=False, autoLevels=False)

        x_axis = np.arange(3564)
        self.curve_h.setData(x_axis, self.halo_h_data)
        self.curve_v.setData(x_axis, self.halo_v_data)

        sum_h = np.sum(self.halo_h_data)
        sum_v = np.sum(self.halo_v_data)

        self.label_sum_h.setText(f"Total Horizontal Halo: {sum_h:.4e}")
        self.label_sum_v.setText(f"Total Vertical Halo: {sum_v:.4e}")

    def closeEvent(self, event):
        self.worker.stop()
        if self.acq:
            self.acq.stop_subscriptions()
        if self.replayer:
            self.replayer.stop()
        event.accept()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online Halo Analysis GUI")
    parser.add_argument("--test-dir", type=str, default=None, help="Directory containing offline data (.pkl.zst) to replay")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    gui = OnlineAnalysisGUI(offline_dir=args.test_dir)
    gui.show()
    sys.exit(app.exec_())
