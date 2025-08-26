import pyjapc
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CoronagraphDevices

japc = pyjapc.PyJapc(selector = '', incaAcceleratorName = '')
japc.rbacLogin(loginDialog=True)

LHC = CoronagraphDevices(japc)

japc.startSubscriptions()
