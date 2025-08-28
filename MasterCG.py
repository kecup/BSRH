## So this is it!
from config import CoronagraphDevices
from devices_lib import Camera, Motor, Pneumatics, FWheel
from aligners_lib import T1T2AlignmentNN, T3T4AlignmentNN, T5T6AlignmentNN
import pyjapc
import pyrbac

# import acquisition_lib
# import ??

# Start with putting all in, then check the position on each camera and align only the piece that is necessary...

#We need 2 japc instances to separate the subscriptions
client = pyrbac.AuthenticationClient()
token = client.login_kerberos()

japcTg = pyjapc.PyJapc(selector = '', incaAcceleratorName = '')
japcTg.rbacLogin(token = token)

japcAcq = pyjapc.PyJapc(selector = '', incaAcceleratorName = '')
japcAcq.rbacLogin(token = token)


# Here we create instances... Targets, Cameras within aligners,
Targets = CoronagraphDevices(japcTg)

