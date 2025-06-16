import pyjapc
import pyrbac
import sys
import select
import dill as pickle
import time
import os
import zstandard as zstd
import datetime as dt
import threading

DATA_PATH_ROOT = '/user/jpucek/BWS/Data'
BWS_V = 'LHC.BWS.5L4.B2V1'
BWS_H = 'LHC.BWS.5L4.B2H1'
cctx = zstd.ZstdCompressor(level=5)
run_number = 0

class DataBWS:
    def __init__(self, name, value, header):
        self.name = name
        self.value = value
        self.header = header

def saveData(name, value, header):
    #This is a function, that will serve as a simple saver of BWS into the NFS
    data = DataBWS(name, value, header)
    if 'V' in name:
        write_chunk(data)
    else:
        write_chunk(data, False)
    print('Saved ', name, ' data to NFS at ', dt.datetime.now())

def write_chunk(events, vertical = True):
    global run_number
    ts = time.strftime('%Y_%m_%d', time.localtime())
    filename = f"{ts}_run_{run_number:02d}_chunk_{int(time.time())}.pkl.zst"
    if vertical:
        path = os.path.join(DATA_PATH_ROOT+'/Vertical', time.strftime('%Y/%m/%d/', time.localtime()))
    else:
        path = os.path.join(DATA_PATH_ROOT+'/Horizontal', time.strftime('%Y/%m/%d/', time.localtime()))

    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    with open(full_path, 'wb') as f:
        with cctx.stream_writer(f) as compressor:
            pickle.dump(events, compressor)

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

def main():
    japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
    client = pyrbac.AuthenticationClient()
    token = client.login_kerberos()
    japc.rbacLogin(token=token)
    #global run_number
    #run_number = get_last_run_number()+1
    print('Current run number = ', run_number)
    japc.subscribeParam(BWS_V+"/Acquisition", saveData, getHeader=True)
    japc.subscribeParam(BWS_H+"/Acquisition", saveData, getHeader=True)

    japc.startSubscriptions()
    while True:
        if select.select([sys.stdin], [], [], 0.0)[0]:
            print("Key pressed! Exiting loop.")
            break
    japc.stopSubscriptions()
    japc.clearSubscriptions()

if __name__ == "__main__":
    main()