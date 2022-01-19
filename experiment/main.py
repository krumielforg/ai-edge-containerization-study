import csv
import subprocess
from datetime import datetime
import time
import os
import sys
import random

import constants as cnst
import hvpm

USER = "ubuntu"
#HOST = "rpi4"
HOST = "rpi3"
monitor_energy = True
NUM_REPETITIONS = 10


class ExperimentRow:
    def __init__(self, row_id, run_id, depl, model, env):
        self.row_id = row_id
        self.run_id = run_id
        self.depl = depl
        self.model = model
        self.env = env

    def __repr__(self):
        return f'row_id={self.row_id} run_id={self.run_id} depl={self.depl} model={self.model} env={self.env}'
    

def load_rows(csv_file):
    rows = []
    with open(csv_file) as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        env = ""
        for csv_row in csv_reader:
            if line_count == 0:
                env = csv_row[0]
                pass
            elif line_count == 1:
                pass
            else:
                row = ExperimentRow(
                    csv_row[0], csv_row[1], csv_row[2], csv_row[3], env)
                rows.append(row)
            line_count += 1
    return rows


def prepare_log_folder():
    logs_path = cnst.get_log_folder_name()
    return logs_path


# num rows = num_repetitions x num_depl x num_model
rows = load_rows("rows_rpi3.csv")
lpath = prepare_log_folder()
random.shuffle(rows)
while rows[0].depl == 'DOCKER':
    random.shuffle(rows)

with open("run_order.txt", "w+") as f:
    for r in rows: 
        f.write(f'{r}\n')



#if monitor_energy:
power_monitor = hvpm.Monitor(voltage=5.5)
#power_monitor.close()
# time.sleep(120) # wait for power up of rpi
count = 1
#sys.exit(0)

for r in rows:        
    print(f'[{count}/120]', r)
    count += 1

    # stat energy monitoring
    if monitor_energy:
        power_monitor.stop_sampling()  # in case for some reason in remains in that state
        power_monitor.start_sampling(f'./powerlogs/{r.row_id}_{r.run_id}_measurements.csv')

        #time.sleep(10)
        #power_monitor.stop_sampling() 
        #sys.exit(0)

    # run a combination of deployment-model from the experiment table
    if r.depl == cnst.Deployment.DOCKER.name:
        script_path = os.path.join(cnst.MOUNT_WORKING_DIR, 'run_docker.py')
        cmd = f'python3 {script_path} {r.depl} {r.model} {r.row_id} {r.run_id} {lpath}'
    elif r.depl == cnst.Deployment.ONNX.name:        
        script_path = os.path.join(cnst.MOUNT_WORKING_DIR, 'measure.py')
        cmd = f'python3 {script_path} {r.depl} {r.model} {r.row_id} {r.run_id} {lpath}'   
    elif r.depl == cnst.Deployment.SCLB.name:
        script_path = os.path.join(cnst.MOUNT_WORKING_DIR, 'measure.py')
        cmd = f'python3 {script_path} {r.depl} {r.model} {r.row_id} {r.run_id} {lpath}'
    else:
        print(r.depl, "not supported.")
        continue

    print('cmd:', cmd)
    p = subprocess.Popen("ssh {user}@{host} {cmd}".format(user=USER, host=HOST, cmd=cmd),
                        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = p.stderr.read()
    p.communicate()
    print('output:', output)
    

    if monitor_energy:
        power_monitor.stop_sampling(set_condition=True)

    # wait 1 min between  runs
    # break
    print("waiting 60s ...")
    time.sleep(60)
