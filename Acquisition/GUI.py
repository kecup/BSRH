# Data acquisition via master camera and parameter subscriptions
# Now subscribe to all devices, cache their latest values, and use camera callback to assemble full events.

import pyjapc
import pyrbac
import time
from datetime import datetime as dt
import datetime
import sys
import math
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
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from queue import Queue
from config import CoronagraphDevices
from devices_lib import Camera, Motor, Pneumatics
from aligners_lib import T1T2AlignmentNN, T3T4AlignmentNN, T5T6AlignmentNN
from TrainNN import FeedbackNN6, FeedbackNN7


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

CAM_TARGETS = {'cam1': [416,222], 'cam2': [316,237], 'cam3': [713,226], 'cam4': [709, 45], 'cam5': [438, 259], 'cam6': [484, 302]}

# Globals
cctx = zstd.ZstdCompressor(level=5)
data_cache = {}             # stores latest values for each parameter
cache_lock = threading.Lock()
event_queue = Queue()       # to queue complete events for background saving
saver_running = threading.Event()
setter_running = threading.Event()
full_auto_running = threading.Event()
camera_arrived = threading.Event()
firstCallback = True
scanning = False
run_number = 0


def distance(p1, p2):
    """
    Return Euclidean distance between two 2D points (a, b).
    Returns -1 if input is invalid.
    """
    try:
        # Ensure input is iterable with exactly 2 numeric elements
        if (
            not all(isinstance(p, (list, tuple)) for p in (p1, p2)) or
            len(p1) != 2 or len(p2) != 2 or
            not all(isinstance(coord, (int, float)) for coord in (*p1, *p2))
        ):
            raise ValueError("Invalid input")

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.hypot(dx, dy)

    except Exception:
        return -1

def autoExposure(image, pulses):
    maximum = np.max(image)
    newPulses = np.round(3800/maximum)*pulses
    return int(newPulses)

class AllData:
    def __init__(self, timestamp, cam_data, device_data):

        self.timestamp = timestamp
        self.cam_data = cam_data
        self.device_data = device_data
# Background saver thread: takes events from queue and writes them in chunks

class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        if text.strip():  # skip empty lines
            self.textWritten.emit(text)

    def flush(self):
        pass  # no-op for compatibility


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
    
    
def setter_worker(japc, path, value, automatic = False):
    print("[Setter] Worker started.")
    japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting2#triggerDecimation', '0') #Take SW trigger only, why does it not work??
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

def is_saturated(image, threshold=4094, frac=0.05):
    total = image.size
    if total == 0:
        return False
    return np.count_nonzero(image >= threshold) >= total * frac

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

def draw_target(ax, x, y, size=10, gap=4, color="white", lw=2):
    """
    Draw a target marker (cross without the middle part) at (x,y).

    size = total half-length of each arm
    gap  = empty space around the center
    """
    # Horizontal arms
    ax.plot([x - size, x - gap], [y, y], color=color, lw=lw)
    ax.plot([x + gap, x + size], [y, y], color=color, lw=lw)

    # Vertical arms
    ax.plot([x, x], [y - size, y - gap], color=color, lw=lw)
    ax.plot([x, x], [y + gap, y + size], color=color, lw=lw)

def saveImage(name, image, center):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(image, cmap="jet", vmin=0, vmax=np.max(image))
    fig.colorbar(im, ax=ax)

    if len(center) == 2:
        x, y = center
        draw_target(ax, x, y, size=20, gap=10, color="white", lw=2)
        ax.text(x + 20, y - 20, f"({int(x)}, {int(y)})", color="white", fontsize=8, ha="left", va="bottom", bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"))

    plt.tight_layout()
    plt.savefig(name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
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
        self.full_align_btn.clicked.connect(self.toggle_full_alignment_clicked)
        self.right_column.addWidget(self.full_align_btn)

        self.cam_align_btn = QPushButton("Main Cam Alignment")
        self.right_column.addWidget(self.cam_align_btn)

        # Row for Full Autonomy + checkbox
        autonomy_row = QHBoxLayout()
        self.full_auto_btn = QPushButton("Full Autonomy")
        self.full_auto_btn.clicked.connect(self.toggle_automatic_clicked)
        font = self.full_auto_btn.font()
        font.setBold(True)
        self.full_auto_btn.setFont(font)
        autonomy_row.addWidget(self.full_auto_btn)

        info_row = QHBoxLayout()
        self.energyLHC_label = QLabel('LHC energy = -1')
        self.totalIntensityB2_label = QLabel('LHC intensity B2 = -1')
        info_row.addWidget(self.energyLHC_label)
        info_row.addWidget(self.totalIntensityB2_label)

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

        sys.stdout = EmittingStream()
        sys.stdout.textWritten.connect(self.log_message)  # This is here to catch all messages from print...

        self.right_column.addLayout(autonomy_row)
        self.right_column.addLayout(info_row)
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
        self.occulter = True
        
        self.bunchSlots = []
        self.bunchIndex = 0

        self.energyLHC = -1 # The two fields will be used in the fully automated regime...
        self.totalIntensityB2 = -1

        # Getting login token
        self.clientRBAC = pyrbac.AuthenticationClient()
        # tokenLife = datetime.timedelta(days = 7)
        # login_master_kerberos does not asign roles properly, so for now we try the kerberos app login...
        self.token = self.clientRBAC.login_kerberos() # Ok this works well, let's renew the token after each fill, that way we'll be safe.... TODO: renew token and login to japcs before each fill...

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
        self.Cam6 = Camera('LHC.BSRH.BTVDC-CAM.MAIN', self.align_japc, False)

        # This is taking care of the automated procedures - subscribes to all important stuff, even if the request is only to align the main camera...
        self.main_japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
        self.main_japc.rbacLogin(token = self.token)

        # Motor LS; x,y occulter; gate, main camera, intensifier; 2 motors for final alignment - depending whether main alignment needs the individual motors or not...
        self.main_japc.subscribeParam('LHC.BCTFR.A6R4.B2/Acquisition#wholeBeamIntensity', self.intensity_callback) # This is the total intensity in B2
        self.main_japc.subscribeParam('LHC.BCCM.B2.A/SummaryAcquisition#beamEnergy', self.energy_callback)

        self.M3X = Motor('LHC.BSRH.FOLDING-H.B2', self.main_japc, 45)  # LHC
        self.M3Y = Motor('LHC.BSRH.FOLDING-V.B2', self.main_japc, 0)  # LHC
        # TODO: Add LS V-UP for closing,
        self.LS = Pneumatics('LHC.BSRH.IN-OUT.B2', self.main_japc, 2) # This is in targets, no? So why is it here??

        self.bgT2 = np.asarray([-1])
        self.bgT4 = np.asarray([-1])
        self.bgT5 = np.asarray([-1])
        self.bgT6 = np.asarray([-1])

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

    def energy_callback(self, name, value):
        self.energyLHC = value

    def intensity_callback(self, name, value):
        self.totalIntensityB2 = value

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
          # Deal with fast saturation (in case of issues..)
          if is_saturated(image['image2D'], 4094, 0.01): # 5% of image is above 4094 we put in additional filter...
            self.Targets.filterCM.setRelPosition(-1)
          if self.bunchIndex+1 < len(self.bunchSlots):
            self.bunchIndex = self.bunchIndex + 1
          else:
            self.bunchIndex = 0
            if self.occulter == False:
                self.Targets.motXCM.setAbsPosition(32567) # There is something wrong!!!
                self.Targets.motXCM.setAbsPosition(10867)
                self.Targets.filterCM.setAbsPosition(3)
                self.occulter = True

          self.acq_japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting1#delaySlots', self.bunchSlots[self.bunchIndex])
        # camera_arrived.set()
        timestamp = header.get('acqStamp', time.time())
        ev = AllData(timestamp, image, snapshot)
        event_queue.put(ev)
        # print(f"Queued event at {timestamp}")
        
    def get_bunches(self):
        # Populate tree with dummy bunches and checkboxes
        self.bunch_tree.clear()
        bunchIntensities = self.acq_japc.getParam('LHC.BCTFR.A6R4.B2/Acquisition#averageBunchIntensities')
        for i in range(len(bunchIntensities)):
          if bunchIntensities[i] > 0:
            #self.bunchSlots.append(i)
            item = QTreeWidgetItem([str(i), ""])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(1, Qt.Checked) # Change here...
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

    def scan_bunches(self, automatic = False):
        global scanning
        if not scanning:
          if not automatic:
            self.get_checked_bunches()
          if self.bunchSlots == []:
            self.log_message('Bunches are empty!')
            return
          
          #Setter and saver must be running for the bunch scan to work - maybe we could use the subscriptions without saving the data...
          if setter_running.is_set() == False:
            self.toggle_setter(automatic = automatic)
          if saver_running.is_set() == False:
            self.toggle_acquisition()

          self.acq_btn.setEnabled(False)
          scanning = True
          self.scan_bunches_btn.setText("Stop bunch scan")
          self.freq_input.setReadOnly(True)
          camera_arrived.set()
          
        else:
            scanning = False
            if setter_running.is_set():
                self.toggle_setter()
            if saver_running.is_set():
                self.toggle_acquisition()
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

    def toggle_setter(self, automatic = False):
        if not self.setting:
            try:
                setter_running.set()
                setter_path = 'LHC.BSRH.GATE-GEN.5L4.B2/Trig#triggerSW'
                setter_value = 2
                self.setter_thread = threading.Thread(
                    target=setter_worker,
                    args=(self.acq_japc, setter_path, setter_value, automatic),
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
        #TODO: Check that the cameras are updating by checking individual image sums?
        chargeB2 = self.align_japc.getParam('LHC.BCTFR.A6R4.B2/Acquisition#wholeBeamIntensity')  # Normalizing by the full machine at 3.5e14
        try:
            timingFactor = np.round(3.5e14 / chargeB2, 2)
        except:
            timingFactor = 1

        self.Cam1.changeExpTime(int(timingFactor * 150))
        self.Cam2.changeExpTime(int(timingFactor * 50))
        self.Cam4.changeExpTime(int(timingFactor * 500))  # With ND4 in and EA open...
        self.Cam5.changeExpTime(int(timingFactor * 300))

        # Ugly setting of the main camera and the intensifier...
        mainTime = timingFactor*60000
        mainPulses = int(mainTime/89.2)
        self.Cam6.changeExpTime(max(20000, min(mainTime, 1e5))) # Only 20 to 100ms
        self.main_japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting1#delaySlots', 0) # We want to start with the beginning...
        self.main_japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting1#pulseLength', 12000) # We want 3000 bunches - TODO: understand why we cannot use all 3564 slots..
        self.main_japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting1#numberOfPulses', mainPulses)

        # Turn on the intensifier unless it is already on...
        powerArray = self.main_japc.getParam('LHC.BSR.POWER-12V-B.B2/Acquisition#channelState')
        IntensifierOn = powerArray[1]
        if IntensifierOn == False: # We must turn it on
            powerArray[1] = 1
            self.main_japc.setParam('LHC.BSR.POWER-12V-B.B2/Setting#channelState', powerArray, checkDims = False)
            time.sleep(2)

        IntensifierComs = self.main_japc.getParam('LHC.BSRH.IMGINT.5L4.B2/Acquisition#hw_com_ok') # TODO: Then all that rests is to make the proper alignment using M3 on the main camera, that way the occulter motion is simpler...
        while IntensifierComs == False:
            self.log_message('Did not connect to the intensifier, sleeping 5 seconds.')
            time.sleep(5)
            IntensifierComs = self.main_japc.getParam('LHC.BSRH.IMGINT.5L4.B2/Acquisition#hw_com_ok')
            #WE need to initialize it:
            #self.main_japc.setParam('LHC.BSRH.IMGINT.5L4.B2/ReopenTTY', True)
        #Now we just set the gain to 3.7:
        self.main_japc.setParam('LHC.BSRH.IMGINT.5L4.B2/Setting#gain', 3.7)
        time.sleep(2)

    def toggle_references_task(self, flags: int = 63):
        self.take_ref_btn.setEnabled(False)
        self.full_align_btn.setEnabled(False)
        self.cam_align_btn.setEnabled(False)
        self.get_bunches_btn.setEnabled(False)
        self.acq_btn.setEnabled(False)
        self.scan_bunches_btn.setEnabled(False)

        self.prepare_cameras()
        self.align_japc.startSubscriptions() # This to get the system alive
        time.sleep(2)
        self.Targets.eaTgt.moveIn() # This is currenly enough

        self.bgT2 = np.asarray(self.Cam2.captureImage(True)) # This is setting image as bg for beam position determination
        self.bgT4 = np.asarray(self.Cam4.captureImage(True))
        self.bgT5 = np.asarray(self.Cam5.captureImage(True))
        self.bgT6 = np.asarray(self.Cam6.captureImage(True))

        #This is stupid...
        ts = time.strftime('%Y/%m/%d', time.localtime())
        ms = time.strftime('%H_%M_%S', time.localtime())
        path = os.path.join(REFERENCES_PATH_ROOT, ts)
        os.makedirs(path, exist_ok=True)

        cam_files = {f"cam{i}": os.path.join(path, f"cam{i}_{ms}") for i in range(1, 7)}
        cam_images = {}
        cam_centers = {}
        time.sleep(2) # Now we use only eaIn... so 2 seconds are enough - Due to the fact that Targets.allIn() is non blocking = 10 seconds


        if (flags & 1) == 1:
            self.log_message('Measuring cam 1')
            cam_images['cam1'] = self.Cam1.captureImage()
            cam_centers['cam1'] = getBeamPosition(cam_images['cam1'])

        if (flags & 2) == 2:
            self.Targets.measureT2()
            self.log_message('Measuring cam 2')
            cam_images['cam2'] = self.Cam2.captureImage()
            if cam_images['cam2'].shape == self.bgT2.shape:
                cam_images['cam2'] = cam_images['cam2'] - self.bgT2
            cam_centers['cam2'] = getBeamPosition(cam_images['cam2'])

        if (flags & 4) == 4:
            self.Targets.measureT3()
            self.log_message('Measuring cam 3')
            cam_images['cam3'] = self.Cam2.captureImage()
            if cam_images['cam3'].shape == self.bgT2.shape:
                cam_images['cam3'] = cam_images['cam3'] - self.bgT2
            cam_centers['cam3'] = getBeamPosition(cam_images['cam3'])

        if (flags & 8) == 8:
            self.Targets.measureT4()
            self.log_message('Measuring cam 4')
            cam_images['cam4'] = self.Cam4.captureImage()
            if cam_images['cam4'].shape == self.bgT4.shape:
                cam_images['cam4'] = cam_images['cam4'] - self.bgT4
            cam_centers['cam4'] = getBeamPosition(cam_images['cam4'])

        if (flags & 16) == 16:
            self.Targets.measureT5()
            self.log_message('Measuring cam 5')
            cam_images['cam5'] = self.Cam5.captureImage()
            if cam_images['cam5'].shape == self.bgT5.shape:
                cam_images['cam5'] = cam_images['cam5'] - self.bgT5
            cam_centers['cam5'] = getBeamPosition(cam_images['cam5'])

        if (flags & 32) == 32:
            if self.Cam6.exposure > 2e4:
                self.log_message('Measuring cam 6')
                self.Targets.measureT6()
                cam_images['cam6'] = np.asarray(self.Cam6.captureImage())
                if cam_images['cam6'].shape == self.bgT6.shape:
                    cam_images['cam6'] = cam_images['cam6'] - self.bgT6
                cam_centers['cam6'] = getBeamPosition(cam_images['cam6'])

        self.log_message('Saving images')

        for key in cam_images:
            saveImage(cam_files[key], cam_images[key], list(cam_centers[key]))
        self.log_message('Images saved')

        self.Targets.apertures_to_references() # TODO: check the implementation
        self.align_japc.stopSubscriptions() # This stops the cameras and targets
        self.take_ref_btn.setEnabled(True)
        self.full_align_btn.setEnabled(True)
        self.cam_align_btn.setEnabled(True)
        self.get_bunches_btn.setEnabled(True)
        self.acq_btn.setEnabled(True)
        self.scan_bunches_btn.setEnabled(True)

        return cam_centers # This is critical in case we need to align on this - btw we could get around moving targets, right?

    def toggle_references_clicked(self):
        self.align_thread = threading.Thread(target=self.toggle_references_task, args=(True, True, True), daemon=True)
        self.align_thread.start()

    def toggle_full_alignment_task(self, flags: int = 7):
        self.take_ref_btn.setEnabled(False)
        self.full_align_btn.setEnabled(False)
        self.cam_align_btn.setEnabled(False)
        self.get_bunches_btn.setEnabled(False)
        self.acq_btn.setEnabled(False)
        self.scan_bunches_btn.setEnabled(False)

        self.prepare_cameras()

        file_name1 = "/user/jpucek/GitLab/bsrh-scripts/Alignment/Models/lhc_t1_t2_v5_model1_cams_targets-steps.pth"
        X_scaler_name1 = "/user/jpucek/GitLab/bsrh-scripts/Alignment/Models/lhc_t1_t2_v5_model1_cams_targets-steps_X_scaler.pkl"
        y_scaler_name1 = "/user/jpucek/GitLab/bsrh-scripts/Alignment/Models/lhc_t1_t2_v5_model1_cams_targets-steps_y_scaler.pkl"

        file_name2 = "/user/jpucek/GitLab/bsrh-scripts/Alignment/Models/isr_t3_t4_v4_filtered_model3_only_cam2_target2-steps.pth"
        X_scaler_name2 = "/user/jpucek/GitLab/bsrh-scripts/Alignment/Models/isr_t3_t4_v4_filtered_model3_only_cam2_target2-steps_X_scaler.pkl"
        y_scaler_name2 = "/user/jpucek/GitLab/bsrh-scripts/Alignment/Models/isr_t3_t4_v4_filtered_model3_only_cam2_target2-steps_y_scaler.pkl"

        file_name3 = "/user/jpucek/GitLab/bsrh-scripts/Alignment/Models/lhc_t6_v1_model1_camMain_target2-steps.pth" # Folding will stay static, so actuating the same M3...
        X_scaler_name3 = "/user/jpucek/GitLab/bsrh-scripts/Alignment/Models/lhc_t6_v1_model1_camMain_target2-steps_X_scaler.pkl"
        y_scaler_name3 = "/user/jpucek/GitLab/bsrh-scripts/Alignment/Models/lhc_t6_v1_model1_camMain_target2-steps_y_scaler.pkl"


        # only cam2 model
        t1_t2_alignment = T1T2AlignmentNN(model_file_name=file_name1, input_size=8, output_size=3, feedback_nn=FeedbackNN7, X_scaler_name=X_scaler_name1, y_scaler_name=y_scaler_name1, japc=self.align_japc, dist_tolerance=10, config=self.Targets)

        t3_t4_alignment = T3T4AlignmentNN(model_file_name=file_name2, input_size=4, output_size=2, feedback_nn=FeedbackNN6, X_scaler_name=X_scaler_name2, y_scaler_name=y_scaler_name2, japc=self.align_japc, dist_tolerance=10, config=self.Targets, isr=False)

        t5_t6_alignment = T5T6AlignmentNN(model_file_name=file_name3, input_size=4, output_size=2, feedback_nn=FeedbackNN6, X_scaler_name=X_scaler_name3, y_scaler_name=y_scaler_name3, japc=self.align_japc, dist_tolerance=5, config=self.Targets, isr=False)

        self.align_japc.startSubscriptions() # Only after all object were initialized...
        time.sleep(2)
        if (flags & 1) == 1:
            self.log_message('Starting alignment on T1T2')
            # t1_t2_alignment.align() # TODO: make this functional (needs better data, or different model...)


        if (flags & 2) == 2:
            self.log_message('Starting alignment on T4')
            t3_t4_alignment.align() # This is easy as the measurement is ready

        if (flags & 4) == 4:
            self.log_message('Starting alignment on T6')
            t5_t6_alignment.align() # We must make sure that Intensifier is ON and properly setup...
        self.log_message('Alignment done')
        self.align_japc.stopSubscriptions()

        self.take_ref_btn.setEnabled(True)
        self.full_align_btn.setEnabled(True)
        self.cam_align_btn.setEnabled(True)
        self.get_bunches_btn.setEnabled(True)
        self.acq_btn.setEnabled(True)
        self.scan_bunches_btn.setEnabled(True)

    def toggle_full_alignment_clicked(self):
        if self.align_thread ==  None:
            self.align_thread = threading.Thread(target=self.toggle_full_alignment_task, daemon=True)
            self.align_thread.start()
            self.log_message('Alignment thread started')
        else:
            self.align_thread.join()
            self.align_thread = threading.Thread(target=self.toggle_full_alignment_task, daemon=True)
            self.align_thread.start()

    def toggle_automatic_task(self):

        self.log_message('Started fully automated bsrh control')
        self.main_japc.startSubscriptions() # This is to get the different energy and intensity
        time.sleep(1) # To make sure subscriptions start...
        refs = False
        aligned = False


        while full_auto_running.is_set(): # This must be set to True by the button and then it must be set to false by the same button to stop...
            # 1 take references, 2 full alignment (Main Cam not ready...), 3 main camera alignment start, 4 acquisition start
            #Let's get a new token in case ours will expire within 12 hours
            if self.token.expiration_time < dt.now() + datetime.timedelta(hours = 1):
                self.token = self.clientRBAC.login_kerberos() # Here we get new token, that we can use to login to used japcs..
                self.main_japc.rbacLogin(token = self.token)
                self.align_japc.rbacLogin(token = self.token)
                self.acq_japc.rbacLogin(token = self.token)

            with cache_lock:
                energy = self.energyLHC
                intensity = self.totalIntensityB2

                self.energyLHC_label.setText('LHC energy = ' + str(np.round(energy, 2)))
                self.totalIntensityB2_label.setText(f'LHC b2 intensity = {intensity:.3e}')

            if energy > 4500 and intensity > 1e9: # This means we are in business
                if refs == False:
                    # Before anything happens, we must make the system safe...
                    self.Targets.allIn()
                    time.sleep(5)
                    self.log_message('Taking initial references')
                    positions = self.toggle_references_task()
                    print(positions)
                    refs = True

                if aligned == False and refs == True:
                    self.log_message('Performing alignment')
                    self.log_message(positions)
                    if positions['cam6'][0] == -1 or positions['cam6'][1] == -1: # no beam on main cam, align on t4 as well
                        self.toggle_full_alignment_task(6)
                        self.log_message('Taking references after alignment')
                        self.toggle_references_task(32+16+8) # T4, T5, T6
                    else:
                        self.toggle_full_alignment_task(4) # This aligns only the main camera
                        self.log_message('Taking references after alignment')
                        self.toggle_references_task(32)
                    aligned = True

                ## Here we could start the main camera alignment thread, and the acquisition thread... (Don't forget that we want to move the occulter... and subtract the bg for the main cam and the occulter is always on )
                if not scanning:
                    self.get_bunches() # This way all will be checked...
                    self.get_checked_bunches() # Important...
                    self.bunchSlots.append(self.bunchSlots[-1]+1) # This is to grab the BG
                    self.main_japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting1#pulseLength', 4) # bunch-by-bunch measurement
                    self.Targets.filterCM.setAbsPosition(3) # This is for the beginning without occulter...

                    #OK, so here we need to get new exposure time (probably not ideal...)
                    self.align_japc.startSubscriptions()
                    time.sleep(2)
                    self.main_japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting1#delaySlots', self.bunchSlots[0])
                    self.main_japc.setParam('LHC.BSRH.IMGINT.5L4.B2/Setting#gain', 3.9)
                    oldPulses = self.main_japc.getParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting1#numberOfPulses')
                    newPulses = autoExposure(self.Cam6.captureImage(), oldPulses)
                    self.main_japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting1#numberOfPulses', newPulses)
                    newTime = int(oldPulses*89.2+100)
                    self.Cam6.changeExpTime(max(20000, min(newTime, 1e5)))

                    #Now get BG!!:
                    self.main_japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting1#delaySlots', self.bunchSlots[-1])
                    self.Cam6.captureImage(True)

                    self.occulter = False # This is important for the setter that will move the occulter in after one full pass of full images...
                    self.align_japc.stopSubscriptions()
                    time.sleep(2)

                    self.scan_bunches(automatic = True) # In this scanning the speed of acq is really slow...
                time.sleep(5) # This means the GUI will be responsive once in 5 seconds...
            else:
                if scanning:
                    self.scan_bunches() # This stops setter, not saver - why??
                refs = False
                aligned = False
                if self.Targets.eaTgt.isIn() == False:
                    self.Targets.allIn() # Making the system safe...

                # AND turn off the intensifier...
                powerArray = self.main_japc.getParam('LHC.BSR.POWER-12V-B.B2/Acquisition#channelState')
                IntensifierOn = powerArray[1]
                if IntensifierOn:  # We must turn it off
                    powerArray[1] = 0
                    self.main_japc.setParam('LHC.BSR.POWER-12V-B.B2/Setting#channelState', powerArray, checkDims=False)
                time.sleep(5)

        # Here we should make sure we exit properly...
        if scanning:
            self.scan_bunches()
        if self.Targets.eaTgt.isIn() == False:
            self.Targets.eaTgt.moveIn()  # Making the system safe...

        self.main_japc.stopSubscriptions()
        self.log_message('Stopped fully automated bsrh control')

    def toggle_automatic_clicked(self):

        if full_auto_running.is_set():
            full_auto_running.clear()
            self.control_thread.join()
            self.full_auto_btn.setText('Full autonomy')
            self.take_ref_btn.setEnabled(True)
            self.full_align_btn.setEnabled(True)
            self.cam_align_btn.setEnabled(True)
            self.get_bunches_btn.setEnabled(True)
            self.acq_btn.setEnabled(True)
            self.scan_bunches_btn.setEnabled(True)

        else:
            self.take_ref_btn.setEnabled(False)
            self.full_align_btn.setEnabled(False)
            self.cam_align_btn.setEnabled(False)
            self.get_bunches_btn.setEnabled(False)
            self.acq_btn.setEnabled(False)
            self.scan_bunches_btn.setEnabled(False)
            full_auto_running.set()
            try:
                self.control_thread = threading.Thread(target=self.toggle_automatic_task, daemon=True)
                self.control_thread.start()
            except:
                print('The control thread was not started!')
            self.full_auto_btn.setText('Stop running automatically')


# Entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AcquisitionGUI()
    gui.show()
    sys.exit(app.exec_())

