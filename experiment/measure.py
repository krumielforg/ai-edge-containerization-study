import csv
import subprocess
import time
import sys
import os

from datetime import datetime
from timeit import default_timer as timer

import constants as cnst
from prepare_datasets import randomize_input_files

def monitor(lpath, mtype, depl, model, row_id, run_id, pid=None):

    if pid == None:
        return

    # cpu/mem monitor - a bash script calling pidstat periodically
    if mtype == cnst.PerfMonitor.CPUMEM:
        subprocess.Popen(
            ['./measure.sh', str(pid), cnst.get_log_name(lpath, mtype, depl, model, row_id, run_id)])

def run_and_measure(depl, model, row_id, run_id, lpath):

    randomize_input_files()

    if not os.path.exists(lpath):
        os.mkdir(lpath)

    # measure exec time of run
    exectime_log = open(cnst.get_log_name(lpath, cnst.PerfMonitor.EXECTIME, depl, model, row_id, run_id), "a+")
    e_start = timer()

    if depl == cnst.Deployment.SCLB.name:
        cmd = [cnst.SCLB_BINARY, cnst.get_aot_path(model), cnst.get_input_path(model), "1", str(cnst.dataset_files[model][1])]
        print(cmd)
        p = subprocess.Popen([cnst.SCLB_BINARY, cnst.get_aot_path(model), cnst.get_input_path(model), "1", str(cnst.dataset_files[model][1])], stdout=subprocess.DEVNULL)
        pid = p.pid     
        monitor(lpath, cnst.PerfMonitor.CPUMEM, depl, model, row_id, run_id, pid=pid)

        try:
            # wait for completion of run
            p.communicate()
        except subprocess.TimeoutExpired as e:
            p.kill()


    else:
        p = subprocess.Popen(['python3', 'run.py', depl, model])
        pid = p.pid
        monitor(lpath, cnst.PerfMonitor.CPUMEM, depl, model, row_id, run_id, pid=pid)

        try:
            # wait for completion of run
            p.communicate()
        except subprocess.TimeoutExpired as e:
            p.kill()
    
    e_end = timer()
    e_time = e_end - e_start
    print(e_time,"s:", e_start, e_end)
    exectime_log.write(cnst.get_log_line(cnst.PerfMonitor.EXECTIME, run_id, time=e_time, start=e_start, end=e_end))
    exectime_log.close()

if __name__ == "__main__":
    os.chdir(cnst.MOUNT_WORKING_DIR)
    run_and_measure(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])


