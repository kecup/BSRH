# gather data on isr - mirror2 (x, y, phi, theta) + target3 (see-through) + target4

import pyjapc
from time import sleep
import csv
import random
import sys
import os
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from devices_lib import Camera, Motor, Pneumatics
import pyrbac