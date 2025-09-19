# Data acquisition via master camera and parameter subscriptions
# Now subscribe to all devices, cache their latest values, and use camera callback to assemble full events.

import pyjapc
import pyrbac
import time
from datetime import datetime as dt
import sys
import os
import threading
import numpy as np
import dill as pickle
import zstandard as zstd
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from PyQt5.QtWidgets import (QApplication, QWidget, QPlainTextEdit, QVBoxLayout, QPushButton, QCheckBox, QSpinBox, QLabel, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QMessageBox, QLineEdit)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt, pyqtSignal
from queue import Queue
from config import CoronagraphDevices
from devices_lib import Camera
from aligners_lib import T1T2AlignmentNN, T3T4AlignmentNN, T5T6AlignmentNN


# Configuration
DEVICE_LIST = {
    'L4': 'LHC.BSRH.L4-Z-TS50.B2/Acquisition#position',
    'L5z': 'LHC.BSRH.L5-Z-TS25.B2/Acquisition#position',
    'L5x':'LHC.BSRH.L5-X-TS25.B2/Acquisition#position',
    'Folding-V':'LHC.BSRH.FOLDING-V.B2/Acquisition#position',
    'Folding-H':'LHC.BSRH.FOLDING-H.B2/Acquisition#position',
    'Collimator-V':'TCP.D6R7.B2/MeasuredCornerPositions',
    'Collimator-H':'TCP.C6R7.B2/MeasuredCornerPositions',
    'FBCT1':'LHC.BCTFR.A6R4.B2/Acquisition',
    'FBCT2':'LHC.BCTFR.B6R4.B2/Acquisition',
    'CM-FWL':'LHC.BSRH.CM-FWL.B2/Acquisition#position', 
    'CMASK-X':'LHC.BSRH.CMASK-X-TSZ100.B2/Acquisition#position',
    'CMASK-Y':'LHC.BSRH.CMASK-Y-TSZ100.B2/Acquisition#position',
    'EMASK-HL':'LHC.BSRH.EMASK-H-LEFT.B2/Acquisition#position',
    'EMASK-HR':'LHC.BSRH.EMASK-H-RIGHT.B2/Acquisition#position',
    'EMASK-VD':'LHC.BSRH.EMASK-V-DOWN.B2/Acquisition#position',
    'EMASK-VU':'LHC.BSRH.EMASK-V-UP.B2/Acquisition#position',
    'FW-COLOR':'LHC.BSRH.FW-COLOR.B2/Acquisition#position',
    'LYOT-HL':'LHC.BSRH.LYOT-H-LEFT.B2/Acquisition#position',
    'LYOT-HR':'LHC.BSRH.LYOT-H-RIGHT.B2/Acquisition#position',
    'LYOT-VU':'LHC.BSRH.LYOT-V-UP.B2/Acquisition#position',
    'LYOT-VD':'LHC.BSRH.LYOT-V-DOWN.B2/Acquisition#position',
    'M1-X':'LHC.BSRH.M1-X-TS50.B2/Acquisition#position',
    'M3-PHI':'LHC.BSRH.M3-PHI.B2/Acquisition#position',
    'M3-THETA':'LHC.BSRH.M3-THETA.B2/Acquisition#position',
    'M3-X':'LHC.BSRH.M3-X-TSZ100.B2/Acquisition#position',
    'M3-Y':'LHC.BSRH.M3-Y-TSZ100.B2/Acquisition#position',
    'M3-Z':'LHC.BSRH.M3-Z-TSZ100.B2/Acquisition#position',
    'STEERING-H':'LHC.BSRH.STEERING-H.B2/Acquisition#position',
    'STEERING-V':'LHC.BSRH.STEERING-V.B2/Acquisition#position',
    'INTENSIFIER':'LHC.BSRH.IMGINT.5L4.B2/Acquisition',
    'GATE-CAM':'LHC.BSRH.GATE-GEN.5L4.B2/Setting2',
    'FILTERS.B2':'LHC.BSRH.FILTERS.B2/Acquisition'
}
CAMERA_DEVICE = 'LHC.BSRH.BTVDC-CAM.MAIN/LastImage'
INTENSIFIER_DEVICE = 'LHC.BSRH.GATE-GEN.5L4.B2/Setting1'
DATA_PATH_ROOT = '/user/jpucek/BSRH/Data'
REFERENCES_PATH_ROOT = '/user/jpucek/BSRH/Data/References'
FILE_SIZE = 100

# Globals
cctx = zstd.ZstdCompressor(level=5)
data_cache = {}             # stores latest values for each parameter
cache_lock = threading.Lock()
event_queue = Queue()       # to queue complete events for background saving
saver_running = threading.Event()
setter_running = threading.Event()
camera_arrived = threading.Event()
firstCallback = True
scanning = False
run_number = 0


class AllData:
    def __init__(self, timestamp, cam_data, device_data):

        self.timestamp = timestamp
        self.cam_data = cam_data
        self.device_data = device_data
# Background saver thread: takes events from queue and writes them in chunks

def saver_worker():
    buffer = []
    print("[Saver] Worker started.")
    while saver_running.is_set() or not event_queue.empty():
        try:
            ev = event_queue.get(timeout=0.5)
            buffer.append(ev)
            if len(buffer) >= FILE_SIZE:
                write_chunk(buffer)
                buffer.clear()
        except:
            continue
    if buffer:
        write_chunk(buffer)
    print("[Saver] Worker exiting.")
    
    
def setter_worker(japc, path, value):
    print("[Setter] Worker started.")
    japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting2#triggerDecimation', '0') #Take SW trigger only
    while setter_running.is_set():
        if not camera_arrived.wait(timeout = 2):
          continue

        try:
          japc.setParam(path,value)
        except Exception as e:
            print(f"[Setter] Error setting parameter: {e}")

        camera_arrived.clear()

    japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting2#triggerDecimation', '11245') #Put back the 1Hz trigger
    print("[Setter] Worker exiting.")


def write_chunk(events):
    global run_number
    ts = time.strftime('%Y_%m_%d', time.localtime())
    filename = f"{ts}_run_{run_number:02d}_chunk_{int(time.time())}.pkl.zst"
    path = os.path.join(DATA_PATH_ROOT, time.strftime('%Y/%m/%d/', time.localtime()))
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    with open(full_path, 'wb') as f:
      with cctx.stream_writer(f) as compressor:
        pickle.dump(events, compressor)
    print(f"Wrote {len(events)} events to {full_path}")

# Utility
def get_last_run_number():
    today_dir = os.path.join(DATA_PATH_ROOT, time.strftime('%Y/%m/%d/', time.localtime()))
    if not os.path.isdir(today_dir): return 0
    files = os.listdir(today_dir)
    runs = []
    for f in files:
        parts = f.split('_run_')
        if len(parts) > 1:
            num = parts[1].split('_')[0]
            if num.isdigit(): runs.append(int(num))
    return max(runs) if runs else 0

def lowLength(array):
    #Identify maxima from a moving average, to smear the hot pixels
    maxima = find_peaks(array, height = 1000, distance = 10, width = 5) # might be empty...

    # We want the highest two of the array:
    try:
        values = np.array(array)[maxima]
        indices = maxima[np.argsort(values)[-2:][::-1]]
    except:
        indices = [0,0]

    return indices[1]-indices[0]

def gaussian(x, amp, cen, wid, const):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2)) + const

def getBeamPosition(image, com = False, occulter = None): #com = Center of Mass, different way of center determination for images that cannot be represented by a Gaussian..., occulter provides list of x,y that is the supposed center of the occulter
    ##Get an image that we will analyse:
    outX = -1
    outY = -1

    xTemp = range(len(image[0]))
    yTemp = range(len(image))

    if occulter != None:
        lengthX = []
        lengthY = []

        # Refine the location
        for offset in range(-5,6): #Well, this might be problematic in case the occulter is not there, yet we perform this...
            crosscutX = image[occulter[0]+offset,:]
            crosscutY = np.transpose(image[:,occulter[1]+offset])
            lengthX.append(lowLength(crosscutX))
            lengthY.append(lowLength(crosscutY))

        # Here we use the one with the longest low, should be the same in both axis:
        Radius = (np.max(lengthX) + np.max(lengthY))/2 + 5

        OCX = occulter[0] + lengthX.index(np.max(lengthX))
        OCY = occulter[1] + lengthY.index(np.max(lengthY))

        xTemp = list(range(0, OCX-Radius))+list(range(OCX+Radius, len(image)))
        yTemp = list(range(0, OCY-Radius))+list(range(OCY+Radius, len(image[0])))


    if com == True:
        center = center_of_mass(image)
        outX = center[1]
        outY = center[0]

    else:
        try:
            projection_x = np.sum(image, axis=0)
            projection_y = np.sum(image, axis=1)
        except:
            projection_x = []
            projection_y = []

        ##Prepare bounds for fits:
        gBounds = ([1e2, 0, 1, 10], [1e7, 960, 100, 2e4])

        try:
            # resX = [np.max(projection_x), np.argmax(projection_x)]
            resX, _ = curve_fit(gaussian, xTemp, projection_x[xTemp],(6e4, np.argmax(projection_x), 10.0, 300.0), bounds=gBounds)
            if resX[0] < 2e3:
                resX[1] = -1
        except:
            print("Did not fit in X.")
            resX = [-1, -1, -1]

        try:
            # resY = [np.max(projection_y), np.argmax(projection_y)]
            resY, _ = curve_fit(gaussian, yTemp, projection_y[yTemp], (6e4, np.argmax(projection_y), 10.0, 300.0), bounds=gBounds)
            if resY[0] < 2e3:
                resY[1] = -1
        except:
            print("Did not fit in Y.")
            resY = [-1, -1, -1]

        outX = resX[1]
        outY = resY[1]

    return np.round(outX,1), np.round(outY,1)


def saveImage(name, image, center):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='jet', vmin=0, vmax=np.max(image)-50)
    plt.colorbar()

    if len(center) == 2:
        plt.plot(center[0], center[1], marker='+', markersize=12, color='white', markeredgewidth=2)
        plt.text(center[0] + 5, center[1] - 5, f"({center[0]}, {center[1]})", color="white", fontsize=10, ha="left", va="bottom", bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"))
    plt.tight_layout()
    plt.savefig(name, dpi=150, bbox_inches="tight")
    plt.close()
    
# GUI Definition
class AcquisitionGUI(QWidget):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.log_signal.connect(self._append_log)

        self.setWindowTitle("Simple Acquisition GUI")
        self.setMinimumSize(500, 300)
        self.main_layout = QHBoxLayout()
        self.left_column = QVBoxLayout()
        self.right_column = QVBoxLayout()

        # Controls layout
        controls_layout = QHBoxLayout()
        self.freq_label = QLabel("Freq(Hz):")
        self.freq_input = QSpinBox()
        self.freq_input.setRange(1, 5)
        self.freq_input.setValue(1)
        self.freq_input.valueChanged.connect(self.on_frequency_change)
        controls_layout.addWidget(self.freq_label)
        controls_layout.addWidget(self.freq_input)

        self.acq_btn = QPushButton("Start Acquisition")
        self.acq_btn.clicked.connect(self.toggle_acquisition)
        controls_layout.addWidget(self.acq_btn)

        # self.main_layout.addLayout(controls_layout)
        self.left_column.addLayout(controls_layout)

        # getBunches button
        self.get_bunches_btn = QPushButton("Get Bunches")
        self.get_bunches_btn.clicked.connect(self.get_bunches)
        self.left_column.addWidget(self.get_bunches_btn)

        # Tree for bunch selection
        self.bunch_tree = QTreeWidget()
        self.bunch_tree.setHeaderLabels(["Bunch ID", "Scan"])
        self.left_column.addWidget(self.bunch_tree)

        self.new_bunch_input = QLineEdit()
        self.new_bunch_input.setPlaceholderText("Enter bunch ID (0-3550)")
        validator = QIntValidator(0, 3550)
        self.new_bunch_input.setValidator(validator)
        self.add_bunch_btn = QPushButton("Add Bunch")
        self.add_bunch_btn.clicked.connect(self.add_bunch)
        h_manual = QHBoxLayout()
        h_manual.addWidget(self.new_bunch_input)
        h_manual.addWidget(self.add_bunch_btn)
        self.left_column.addLayout(h_manual)
        
        self.scan_bunches_btn = QPushButton("Scan bunches")
        self.scan_bunches_btn.clicked.connect(self.scan_bunches)
        self.left_column.addWidget(self.scan_bunches_btn)


        self.take_ref_btn = QPushButton("Take References")
        self.take_ref_btn.clicked.connect(self.toggle_references_clicked)
        self.right_column.addWidget(self.take_ref_btn)

        self.full_align_btn = QPushButton("Full Alignment")
        self.right_column.addWidget(self.full_align_btn)

        self.cam_align_btn = QPushButton("Main Cam Alignment")
        self.right_column.addWidget(self.cam_align_btn)

        # Row for Full Autonomy + checkbox
        autonomy_row = QHBoxLayout()
        self.full_auto_btn = QPushButton("Full Autonomy")
        font = self.full_auto_btn.font()
        font.setBold(True)
        self.full_auto_btn.setFont(font)
        autonomy_row.addWidget(self.full_auto_btn)

        self.save_full_checkbox = QCheckBox("save full images")
        self.save_full_checkbox.setToolTip(
            "Switch between saving full images (storage intensive) and saving profiles"
        )
        autonomy_row.addWidget(self.save_full_checkbox)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Logs will appear here...")
        self.right_column.addWidget(QLabel("Log messages:"))
        self.right_column.addWidget(self.log_box)

        self.right_column.addLayout(autonomy_row)
        self.right_column.addStretch()

        # Add right column to root layout
        # self.main_layout.addLayout(self.right_column)

        self.main_layout.addLayout(self.left_column)
        self.main_layout.addLayout(self.right_column)

        self.setLayout(self.main_layout)

        self.acquiring = False
        self.setting = False
        self.saver_thread = None
        self.setter_thread = None
        self.align_thread = None
        self.control_thread = None
        
        self.bunchSlots = []
        self.bunchIndex = 0

        self.energyLHC = -1 # The two fields will be used in the fully automated regime...
        self.totalIntensityB2 = -1

        # Getting login token
        self.clientRBAC = pyrbac.AuthenticationClient()
        self.token = self.clientRBAC.login_kerberos()

        # This is taking care of acquisition
        self.acq_japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
        self.acq_japc.rbacLogin(token = self.token)
        # Subscribe to all device parameters
        for alias, path in DEVICE_LIST.items():
            self.acq_japc.subscribeParam(path, lambda name, value, a=alias: self.generic_callback(a, value))
        # Subscribe to camera with header
        self.acq_japc.subscribeParam(CAMERA_DEVICE, self.camera_callback, getHeader=True)
        self.acq_japc.subscribeParam(INTENSIFIER_DEVICE, self.intensifier_callback, getHeader = False)

        # This is taking care of alignment
        self.align_japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
        self.align_japc.rbacLogin(token = self.token)

        self.Targets = CoronagraphDevices(self.align_japc)

        self.Cam1 = Camera('LHC.BSRH.MON.EA.B2', self.align_japc)
        self.Cam2 = Camera('LHC.BSRH.MON.TGTM3.B2', self.align_japc)
        self.Cam4 = Camera('LHC.BSRH.CAM.CM.B2', self.align_japc)
        self.Cam5 = Camera('LHC.BSRH.MON.LS.B2', self.align_japc)

        # This is taking care of the automated procedures - subscribes to all important stuff, even if the request is only to align the main camera...
        self.main_japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
        self.main_japc.rbacLogin(token = self.token)

        # Motor LS; x,y occulter; gate, main camera, intensifier; 2 motors for final alignment - depending whether main alignment needs the individual motors or not...
        self.main_japc.subscribeParam('LHC.BCTFR.A6R4.B2/Acquisition#wholeBeamIntensity', self.example_callback) # This is the total intensity in B2
        self.main_japc.subscribeParam('LHC.BCCM.B2.A/SummaryAcquisition#beamEnergy', self.example_callback) # TODO: change example callback to a real one...


        self.bgT2 = np.asarray([-1])
        self.bgT4 = np.asarray([-1])
        self.bgT5 = np.asarray([-1])

   # Subscription callbacks

    def generic_callback(self, name, value):
        # store latest value in cache
        with cache_lock:
            data_cache[name] = value

    def intensifier_callback(self, name, value):
        if scanning:
          ##Here sending trigger
          camera_arrived.set()
        with cache_lock:
            data_cache['GATE-INT'] = value

    def camera_callback(self, name, image, header):
        global firstCallback, scanning
                
        if firstCallback:
          firstCallback = False
          return
        # on camera trigger, assemble event from cache snapshot plus image
        with cache_lock:
            snapshot = data_cache.copy()
        #In case of scanning bunches, we change the delaySlots after we make the copy which slows everything down... Could we think about making it faster? The camera could acquire images during the saving time...
        if scanning:
          if self.bunchIndex+1 < len(self.bunchSlots):
            self.bunchIndex = self.bunchIndex + 1
          else:
            self.bunchIndex = 0
          self.acq_japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting1#delaySlots', self.bunchSlots[self.bunchIndex])
        #camera_arrived.set()
        timestamp = header.get('acqStamp', time.time())
        ev = AllData(timestamp, image, snapshot)
        event_queue.put(ev)
        print(f"Queued event at {timestamp}")
        
    def get_bunches(self):
        # Populate tree with dummy bunches and checkboxes
        self.bunch_tree.clear()
        bunchIntensities = self.acq_japc.getParam('LHC.BCTFR.A6R4.B2/Acquisition#averageBunchIntensities')
        for i in range(len(bunchIntensities)):
          if bunchIntensities[i] > 0:
            #self.bunchSlots.append(i)
            item = QTreeWidgetItem([str(i), ""])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(1, Qt.Unchecked)
            self.bunch_tree.addTopLevelItem(item)
            
            
    def add_bunch(self):
        text = self.new_bunch_input.text().strip()
        if text == "":
            return
        item = QTreeWidgetItem([text, ""])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(1, Qt.Checked)
        self.bunch_tree.addTopLevelItem(item)
        self.new_bunch_input.clear()

    def get_checked_bunches(self):
      checked = []
      root = self.bunch_tree.invisibleRootItem()
      for i in range(root.childCount()):
          item = root.child(i)
          if item.checkState(1) == Qt.Checked:
              try:
                  checked.append(int(item.text(0)))
              except ValueError:
                  continue
      self.bunchSlots = checked
      return checked
    def on_frequency_change(self, value):
        triggerDecimation = int(11245/value)
        self.acq_japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting2#triggerDecimation', triggerDecimation)

    def scan_bunches(self):
        global scanning
        if not scanning:
          self.get_checked_bunches()
          if self.bunchSlots == []:
            self.log_message('Bunches are empty!')
            return
          
          #Setter and saver must be running for the bunch scan to work - maybe we could use the subscriptions without saving the data...
          if setter_running.is_set() == False:
            self.toggle_setter()
          if saver_running.is_set() == False:
            self.toggle_acquisition()

          self.acq_btn.setEnabled(False)
          scanning = True
          self.scan_bunches_btn.setText("Stop bunch scan")
          self.freq_input.setReadOnly(True)
          camera_arrived.set()
          
        else:
          scanning = False
          self.toggle_setter()
          self.freq_input.setReadOnly(False)
          self.acq_btn.setEnabled(True)
          self.scan_bunches_btn.setText("Scan bunches")

    def _append_log(self, msg: str):
        ts = dt.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_box.appendPlainText(f"[{ts}] {msg}")

    def log_message(self, msg: str):
        self.log_signal.emit(msg)
        
    def toggle_acquisition(self):
        global saver_running, run_number, firstCallback
        if not self.acquiring:
            try:
                # Start subscriptions and saver thread
                run_number = get_last_run_number() + 1
                self.acq_japc.startSubscriptions()
                saver_running.set()
                self.saver_thread = threading.Thread(target=saver_worker, daemon=True)
                self.saver_thread.start()
                self.acq_btn.setText("Stop Acquisition")
                self.acquiring = True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start:\n{e}")
        else:
            try:
                # Stop subscriptions and saver thread
                self.acq_japc.stopSubscriptions()
                firstCallback = True
                saver_running.clear()
                if self.saver_thread:
                    self.saver_thread.join(timeout=10)
                self.acq_btn.setText("Start Acquisition")
                self.acquiring = False
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to stop:\n{e}")
    
                
    def toggle_setter(self):
        if not self.setting:
            try:
                setter_running.set()
                setter_path = 'LHC.BSRH.GATE-GEN.5L4.B2/Trig#triggerSW'
                setter_value = 2
                self.setter_thread = threading.Thread(
                    target=setter_worker,
                    args=(self.acq_japc, setter_path, setter_value),
                    daemon=True
                )
                self.setter_thread.start()
                self.setting = True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start setter:\n{e}")
        else:
            try:
                setter_running.clear()
                if self.setter_thread:
                    self.setter_thread.join(timeout=2)
                self.setting = False
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to stop setter:{e}")
    
                
    def closeEvent(self, event):
        # Stop acquisition if running
        if self.acquiring:
            try:
                self.acq_japc.stopSubscriptions()
                saver_running.clear()
                if self.saver_thread:
                    self.saver_thread.join(timeout=5)
            except:
                pass
        # Stop setter if running
        if self.setting:
            try:
                setter_running.clear()
                if self.setter_thread:
                    self.setter_thread.join(timeout=2)
            except:
                pass
        event.accept()


    # So here we make the automatic functions:

    def prepare_cameras(self):
        chargeB2 = self.align_japc.getParam('LHC.BCTFR.A6R4.B2/Acquisition#wholeBeamIntensity')  # Normalizing by the full machine at 3.5e14
        try:
            timingFactor = np.round(3.5e14 / chargeB2, 2)
        except:
            timingFactor = 1

        self.Cam1.changeExpTime(timingFactor * 150)
        self.Cam2.changeExpTime(timingFactor * 50)
        self.Cam4.changeExpTime(timingFactor * 500)  # With ND4 in...
        self.Cam5.changeExpTime(timingFactor * 300)

        time.sleep(5)

    def toggle_references_task(self):
        self.take_ref_btn.setEnabled(False)

        self.prepare_cameras()
        self.align_japc.startSubscriptions() # This to get the system alive
        time.sleep(2)
        self.Targets.allIn()

        self.bgT2 = np.asarray(self.Cam2.captureImage(True)) # This is setting image as bg for beam position determination
        self.bgT4 = np.asarray(self.Cam4.captureImage(True))
        self.bgT5 = np.asarray(self.Cam5.captureImage(True))

        ts = time.strftime('%Y_%m_%d', time.localtime())
        path = os.path.join(REFERENCES_PATH_ROOT, time.strftime('%Y/%m/%d/', time.localtime()))
        os.makedirs(path, exist_ok=True)

        cam1_tstamp = 'cam1_' + time.strftime('%H_%M_%S.png', time.localtime())
        cam2_tstamp = 'cam2_' + time.strftime('%H_%M_%S.png', time.localtime())
        cam3_tstamp = 'cam3_' + time.strftime('%H_%M_%S.png', time.localtime())
        cam4_tstamp = 'cam4_' + time.strftime('%H_%M_%S.png', time.localtime())
        cam5_tstamp = 'cam5_' + time.strftime('%H_%M_%S.png', time.localtime())

        cam1_file = os.path.join(path, cam1_tstamp)
        cam2_file = os.path.join(path, cam2_tstamp)
        cam3_file = os.path.join(path, cam3_tstamp)
        cam4_file = os.path.join(path, cam4_tstamp)
        cam5_file = os.path.join(path, cam5_tstamp)

        time.sleep(5) # Due to the fact that Targets.allIn() is non blocking...

        self.log_message('Measuring cam 1')
        cam1_image = self.Cam1.captureImage()
        x1, y1 = getBeamPosition(cam1_image)
        self.Targets.measureT2()

        self.log_message('Measuring cam 2')
        cam2_image = np.asarray(self.Cam2.captureImage())
        if cam2_image.shape == self.bgT2.shape:
            cam2_image = cam2_image - self.bgT2
        x2, y2 = getBeamPosition(cam2_image)
        self.Targets.measureT3()

        self.log_message('Measuring cam 3')
        cam3_image = np.asarray(self.Cam2.captureImage())
        if cam3_image.shape == self.bgT2.shape:
            cam3_image = cam3_image - self.bgT2
        x3, y3 = getBeamPosition(cam3_image)
        self.Targets.measureT4()

        self.log_message('Measuring cam 4')
        cam4_image = np.asarray(self.Cam4.captureImage())
        if cam4_image.shape ==  self.bgT4.shape:
            cam4_image = cam4_image - self.bgT4
        x4, y4 = getBeamPosition(cam4_image)
        self.Targets.measureT5()

        self.log_message('Measuring cam 5')
        cam5_image = np.asarray(self.Cam5.captureImage())
        if cam5_image.shape ==  self.bgT5.shape:
            cam5_image = cam5_image - self.bgT5
        x5, y5 = getBeamPosition(cam5_image)

        self.log_message('Saving images')

        saveImage(cam1_file, cam1_image, [x1, y1])  # There is no bg subtraction for this one...
        saveImage(cam2_file, cam2_image, [x2, y2])
        saveImage(cam3_file, cam3_image, [x3, y3])
        saveImage(cam4_file, cam4_image, [x4, y4])
        saveImage(cam5_file, cam5_image, [x5, y5])
        self.log_message('Images saved')

        self.Targets.apertures_to_references()
        self.align_japc.stopSubscriptions() # This stops the cameras and targets
        self.take_ref_btn.setEnabled(True)

    def toggle_references_clicked(self):
        self.align_thread = threading.Thread(target=self.toggle_references_task, daemon=True)
        self.align_thread.start()

    def toggle_automatic_task(self):
        pass

# Entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AcquisitionGUI()
    gui.show()
    sys.exit(app.exec_())

