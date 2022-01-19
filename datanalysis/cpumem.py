import csv
import os
import sys
from collections import OrderedDict
import statistics as st
import numpy as np
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta

#LOGS_RPI3 = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/logs_rpi3_take2/perflogs"
#LOGS_RPI4 = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/logs_rpi4/perflogs"
LOGS_RPI3 = "../raw_logs/logs_rpi3/perflogs"
LOGS_RPI4 = "../raw_logs/logs_rpi4/perflogs"

#REF_ROWS_RPI4 = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/rows.csv"
#REF_ROWS_RPI3 = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/rows_rpi3.csv"
REF_ROWS_RPI4 = "../experiment/rows.csv"
REF_ROWS_RPI3 = "../experiment/rows_rpi3.csv"

RPI3_MEM_RESULTS = "./logs/mem_rpi3.csv"
RPI4_MEM_RESULTS = "./logs/mem_rpi4.csv"
RPI3_CPU_RESULTS = "./logs/cpu_rpi3.csv"
RPI4_CPU_RESULTS = "./logs/cpu_rpi4.csv"
RPI3_EXEC_RESULTS = "./logs/exec_rpi3.csv"
RPI4_EXEC_RESULTS = "./logs/exec_rpi4.csv"

RPI3_AVG_MEM = "./logs/mem_avg_rpi3.csv"
RPI4_AVG_MEM = "./logs/mem_avg_rpi4.csv"
RPI3_AVG_CPU = "./logs/cpu_avg_rpi3.csv"
RPI4_AVG_CPU = "./logs/cpu_avg_rpi4.csv"
RPI3_AVG_EX = "./logs/exec_avg_rpi3.csv"
RPI4_AVG_EX = "./logs/exec_avg_rpi4.csv"

AC3 = "./logs/all_cpu_rpi3.csv"
AM3 = "./logs/all_mem_rpi3.csv"

AC4 = "./logs/all_cpu_rpi4.csv"
AM4 = "./logs/all_mem_rpi4.csv"


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

def get_row_id(rows, depl, model, run_id):
    for r in rows:
        if r.run_id == run_id and r.depl == depl and r.model == model:
            return r.row_id

def get_start_end(timestamp, start, end):

    new_end = datetime.strptime(timestamp, '%m%d%Y_%H:%M:%S:%f')
    delta = float(end) - float(start)
    new_start = new_end - timedelta(seconds=delta)
    return new_start, new_end

def parse_exec(file_path, rows, depl, model):
    numeric_data = []
    data = {}

    with open(file_path, "r") as f:
        for line in f.readlines():
            val = line.split(" ")[3].split(":")[1]
            numeric_data.append(float(val))

            run_id = line.split(" ")[1].split(":")[1]
            row_id = get_row_id(rows, depl, model, run_id)
            start, end = get_start_end(line.split(" ")[0],
                                        line.split(" ")[4].split(":")[1],
                                        line.split(" ")[5].split(":")[1])

            data[row_id] = {}
            data[row_id]["run_id"] = run_id 
            data[row_id]["model"] = model
            data[row_id]["depl"] = depl
            data[row_id]["exec_time"] = float(val)
            data[row_id]["start"] = start
            data[row_id]["end"] = end
            
    return numeric_data, data

def parse_cpumem(file_path, rows, run_id, depl, model, m_writer, c_writer):
    row_id = get_row_id(rows, depl, model, run_id)

    cpu_data = {}
    mem_data = {}

    pcpu = 0
    tcpu = 0
    rss = 0
    vsz = 0

    with open(file_path, "r") as f:

        lines = f.readlines()[2:]
        count = 0

        t_base = 0
        p_base = 0

        for ln in range(len(lines)):

            # pidstat: TIMESTAMP Time UID PID %usr %system %guest %wait %CPU CPU minflt/s majflt/s VSZ RSS %MEM Command
            # top: TIMESTAMP PID USER PR NI VIRT RES SHR S %CPU %MEM TIME+ COMMAND
            if ln % 2 == 0:
                try:
                    
                    t_line = lines[ln]
                    p_line = lines[ln + 1]

                    p_tokens = p_line.split(" ")
                    t_tokens = t_line.split(" ")

                    # print(p_tokens)
                    # print(t_tokens)

                    # computing time for scatter plot
                    t_timestamp = t_tokens[0] + " " + t_tokens[1]
                    p_timestamp = p_tokens[0] + " " + p_tokens[1]

                    t_time = datetime.strptime(t_timestamp, '%m/%d/%y %H:%M:%S.%f')
                    p_time = datetime.strptime(p_timestamp, '%m/%d/%y %H:%M:%S.%f')

                    if count == 0:
                        t_base = t_time
                        p_base = p_time

                    t_delta = (t_time - t_base).total_seconds()
                    p_delta = (p_time - p_base).total_seconds()

                    pcpu += float(p_tokens[9])
                    rss += int(p_tokens[13])
                    vsz += int(p_tokens[14])
                    tcpu += float(t_tokens[10])

                    m_data = [depl, model,  int(p_tokens[13]), int(p_tokens[14]), p_timestamp, p_delta]
                    m_writer.writerow(m_data)

                    c_data = [depl, model, float(p_tokens[9]), float(t_tokens[10]), p_timestamp, t_timestamp, p_delta, t_delta]
                    c_writer.writerow(c_data)

                    count += 1

                except Exception as e:
                    #print("Warning:", e)
                    continue
            
    #print(rss, vsz)
    #sys.exit(0)
    cpu_data["run_id"] = run_id
    cpu_data["depl"] = depl
    cpu_data["model"] = model
    cpu_data["pcpu"] = pcpu / count
    cpu_data["tcpu"] = tcpu / count

    mem_data["run_id"] = run_id
    mem_data["depl"] = depl
    mem_data["model"] = model
    mem_data["rss"] = (rss / count) / 1000
    mem_data["vsz"] = (vsz / count) / 1000

    return row_id, cpu_data, mem_data

def parse_logs(logs_folder, ecao, mcao, ccao, mco, cco, eco, rf, AC, AM):
    
    rows = load_rows(rf)

    exec_csv_avg_out = open(ecao, 'w')
    mem_csv_avg_out = open(mcao, 'w')
    cpu_csv_avg_out = open(ccao, 'w')
    exec_csv_out = open(eco, "w")
    mem_csv_out = open(mco, "w")
    cpu_csv_out = open(cco, "w")
    ac_out = open(AC, "w")
    am_out = open(AM, "w")
    
    w_ecao = csv.writer(exec_csv_avg_out)
    w_mcao = csv.writer(mem_csv_avg_out)
    w_ccao = csv.writer(cpu_csv_avg_out)
    w_eco = csv.writer(exec_csv_out)
    w_mco = csv.writer(mem_csv_out)
    w_cco = csv.writer(cpu_csv_out)
    w_ac = csv.writer(ac_out)
    w_am = csv.writer(am_out)

    w_eco.writerow(["row_id", "run_id", "depl", "model", "exec_time(s)", "start", "end"])
    w_ecao.writerow(["depl", "model", "mean_exec(s)", "median_exec(s)", "stdev_exec"])

    w_cco.writerow(["row_id", "run_id", "depl", "model", "pcpu ", "tcpu"])
    w_ccao.writerow(["depl", "model", "mean_pcpu(%)", "median_pcpu", "stdev_pcpu", "mean_tcpu(%)", "median_tcpu", "stdev_tcpu"])

    w_mco.writerow(["row_id", "run_id", "depl", "model", "rss(kb)", "vsz(kb)"])
    w_mcao.writerow(["depl", "model", "mean_rss(kb)", "median_rss", "stdev_rss", "mean_vsz(kb)", "median_vsz", "stdev_vsz"])

    w_ac.writerow(["depl", "model", "pcpu", "tcpu", "p_timestamp", "t_timestamp", "p_time", "t_time"])
    w_am.writerow(["depl", "model", "rss(kb)", "vsz(kb)", "timestamp", "time"])

    all_exec_data = {}
    all_cpu_data = {}
    all_mem_data = {}

    for file in os.listdir(logs_folder):
        if not file.endswith('log.txt'):
            continue
        print(file)
        tokens=file.split("_")
        if tokens[0].isnumeric():
            # 1_DOCKER_cifar10_11222021_14:57:38:491940_log.txt
            run_id = tokens[0]
            depl = tokens[1]
            model = tokens[2]
            row_id, cpu_data, mem_data = parse_cpumem(os.path.join(logs_folder, file), rows, run_id, depl, model, w_am, w_ac)
            all_cpu_data[row_id] = cpu_data
            all_mem_data[row_id] = mem_data
        else:
            depl = tokens[0]
            model = tokens[1]
            numeric_data, exec_data = parse_exec(os.path.join(logs_folder, file), rows, depl, model)
            all_exec_data.update(exec_data)
            csv_line = [depl, model, st.mean(numeric_data), st.median(numeric_data), st.stdev(numeric_data)]
            w_ecao.writerow(csv_line)

    


    # some debug code
    '''
    print(all_cpu_data)
    #print(all_exec_data)
    #print(all_mem_data)
    test_list = list(all_cpu_data.keys())
    num_list = []
    for i in range(0, len(test_list)) :
        if test_list[i] is None:
            continue
        num_list.append(int(test_list[i]))
    num_list.sort()
    
    for i in num_list:
        print(i)
    print (len(list(all_cpu_data.keys()))==len(set(list(all_cpu_data.keys()))))
    '''

    # write cpu
    p_data = {}
    t_data = {}
    all_data_sorted = OrderedDict(sorted(all_cpu_data.items(), key=lambda t: int(t[0])))
    for i in all_data_sorted.items():
        csv_line = [i[0], i[1]["run_id"], i[1]["depl"], i[1]["model"],
                        i[1]["pcpu"], i[1]["tcpu"]]

        w_cco.writerow(csv_line)
        
        index = int(i[0]) % 12
        if index == 0:
            index = 12
        if int(i[0]) <= 12:
            p_data[index] = [float(i[1]["pcpu"])]
            t_data[index] = [float(i[1]["tcpu"])]
        elif int(i[0]) >= 109:
            p_data[index].append(float(i[1]["pcpu"]))
            t_data[index].append(float(i[1]["tcpu"]))

            row = [i[1]["depl"], i[1]["model"],
                   st.mean(p_data[index]), st.median(p_data[index]), st.stdev(p_data[index]),
                   st.mean(t_data[index]), st.median(t_data[index]), st.stdev(t_data[index])]
            
            w_ccao.writerow(row)

    # write mme
    r_data = {}
    v_data = {}
    all_data_sorted = OrderedDict(sorted(all_mem_data.items(), key=lambda t: int(t[0])))
    for i in all_data_sorted.items():
        csv_line = [i[0], i[1]["run_id"], i[1]["depl"], i[1]["model"],
                        i[1]["rss"], i[1]["vsz"]]

        w_mco.writerow(csv_line)

        index = int(i[0]) % 12
        if index == 0:
            index = 12
        if int(i[0]) <= 12:
            r_data[index] = [float(i[1]["rss"])]
            v_data[index] = [float(i[1]["vsz"])]
        elif int(i[0]) >= 109:
            r_data[index].append(float(i[1]["rss"]))
            v_data[index].append(float(i[1]["vsz"]))

            row = [i[1]["depl"], i[1]["model"],
                   st.mean(r_data[index]), st.median(r_data[index]), st.stdev(r_data[index]),
                   st.mean(v_data[index]), st.median(v_data[index]), st.stdev(v_data[index])]
            
            w_mcao.writerow(row)


    # write exec
    all_data_sorted = OrderedDict(sorted(all_exec_data.items(), key=lambda t: int(t[0])))
    for i in all_data_sorted.items():
        csv_line = [i[0], i[1]["run_id"], i[1]["depl"], i[1]["model"],
                        i[1]["exec_time"], i[1]["start"], i[1]["end"]]

        w_eco.writerow(csv_line)

parse_logs(LOGS_RPI4, RPI4_AVG_EX, RPI4_AVG_MEM, RPI4_AVG_CPU, RPI4_MEM_RESULTS, RPI4_CPU_RESULTS, RPI4_EXEC_RESULTS, REF_ROWS_RPI4, AC4, AM4)
parse_logs(LOGS_RPI3, RPI3_AVG_EX, RPI3_AVG_MEM, RPI3_AVG_CPU, RPI3_MEM_RESULTS, RPI3_CPU_RESULTS, RPI3_EXEC_RESULTS, REF_ROWS_RPI3, AC3, AM3)
