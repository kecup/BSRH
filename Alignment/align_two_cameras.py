import pyjapc
import pyrbac
import numpy as np
from time import sleep
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import select
from devices_lib import Camera, Motor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


japc = pyjapc.PyJapc(selector = '', incaAcceleratorName = '')
client = pyrbac.AuthenticationClient()
token = client.login_kerberos()
japc.rbacLogin(token=token)
