import csv
import os
from collections import OrderedDict
import statistics as st
import numpy as np
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta


#OWERLOGS_RPI3 = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/logs_rpi3_take2/powerlogs"
#POWERLOGS_RPI4 = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/logs_rpi4/powerlogs"
POWERLOGS_RPI3 = "../raw_logs/logs_rpi3/powerlogs"
POWERLOGS_RPI4 = "../raw_logs/logs_rpi4/powerlogs"

#REF_ROWS_RPI4 = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/rows.csv"
#REF_ROWS_RPI3 = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/rows_rpi3.csv"
REF_ROWS_RPI4 = "../experiment/rows.csv"
REF_ROWS_RPI3 = "../experiment/rows_rpi3.csv"


RPI3_ENERGY_RESULTS = "./logs/energy_rpi3.csv"
RPI4_ENERGY_RESULTS = "./logs/energy_rpi4.csv"

RPI3_AVG = "./logs/energy_avg_rpi3.csv"
RPI4_AVG = "./logs/energy_avg_rpi4.csv"

ALL_RPI3 = "./logs/all_energy_rpi3.csv"
ALL_RPI4 = "./logs/all_energy_rpi4.csv"

EXEC_4 = "./logs/exec_rpi4.csv"
EXEC_3 = "./logs/exec_rpi3.csv"


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


def compute_average_energy(n, current, time, voltage=5):
    # E = P(t) * Δt (Δt = tn - t0)
    # P(t) = V * I(t))
    # I(t) = sum(k=1,n): (Ik-1 + Ik) * (tk - tk-1)/2 - area of the trapezoids from graph

    # print(n)
    It = 0
    delta_t = time[n - 1] - time[0]
    intermediate_values = []

    for k in range(1, n):
        It += ((current[k-1] + current[k]) * (time[k] - time[k-1])) / 2
        
        #print(k, It * voltage * time[k], time[k])
        intermediate_values.append((k, It * voltage * time[k], time[k]))

    Pt = voltage * It
    E = Pt * delta_t

    return E, delta_t, intermediate_values

def get_exec_time(exec_ref, row_id):
    with open(exec_ref, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == row_id:
                start = row[5]
                end = row[6]
                
                datetime_start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S.%f') # .replace(microsecond=0)
                datetime_end = datetime.strptime(end, '%Y-%m-%d %H:%M:%S.%f') # .replace(microsecond=0)

                return datetime_start, datetime_end, row[4]   
            
    return None, None, None

def read_csv_into_dict(row_id, file_path, exec_ref):
    #print(file_path)

    data = {}
    data["time"] = []
    data["current"] = []
    data["voltage"] = []

    count = 0
    first = True
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            count += 1
            if count == 1:
                continue
            if count == 2:
                ref_time = int(row[0])
                continue

            # check if time is in time frame of execution 
            hvpm_time = datetime.fromtimestamp(int(row[0])) - timedelta(hours=1)
            start_exec_time, end_exec_time, _ = get_exec_time(exec_ref, row_id)

        
            if hvpm_time < start_exec_time or hvpm_time > end_exec_time:
                '''
                if hvpm_time < start_exec_time:
                    #pass
                    if row_id == "22":
                        #print("small")
                        print(row_id, "small", hvpm_time, start_exec_time, end_exec_time)
                if hvpm_time > end_exec_time:
                    #pass
                    if row_id == "22":
                        #print("big")
                        print(row_id, "big", hvpm_time, start_exec_time, end_exec_time)
                '''
                continue
            else:
                if first:
                    ref_time = int(row[0])
                    first = False

            #if row_id == "22":
            #    print(row_id, hvpm_time, start_exec_time, end_exec_time)
            
            # print("ok")

            data["time"].append(int(row[0]) - ref_time)  # seconds
            data["current"].append(float(row[1]) / 1000)  # convert mA to A
            data["voltage"].append(float(row[2]))


    unique_time = list(set(data["time"]))
    
    #if row_id == "22":
    #    print(unique_time)
    
    unique_current = []
    # print(len(unique_time))
    for i in range(0, len(unique_time)):
        curr_sum = 0
        curr_sum_count = 0
        for v in range(len(data["time"])):
            if unique_time[i] == data["time"][v]:
                curr_sum += data["current"][v]
                curr_sum_count += 1
        unique_current.append(float(curr_sum/curr_sum_count))

    # print(unique_current)
    # return data

    data["time"] = unique_time
    data["current"] = unique_current

    #if row_id == "22":
    #    print(data)
    
    return data


def get_reference_row(rows, row_id):
    for r in rows:
        if r.row_id == row_id:
            return r


def write_eng(ref_file, eng_file, avg_file, all_file, logs, exec_ref):
    measures = {}

    rows = load_rows(ref_file)

    csv_out = open(eng_file, 'w')
    csv_avg_out = open(avg_file, 'w')
    csv_all_out = open(all_file, 'w')

    writer = csv.writer(csv_out)
    writer_avg = csv.writer(csv_avg_out)
    writer_all = csv.writer(csv_all_out)

    writer.writerow(["row_id", "run_id", "depl", "model", "energy(joules)", "time(seconds)"])
    writer_avg.writerow(["depl", "model", "mean_e(j)", "median_e(j)", "stdev_e", "mean_t(s)", "median_t(s)", "stdev_t"])
    writer_all.writerow(["depl","model","energy(joules)","timestamp"])

    for logfile in os.listdir(logs):
        tokens = logfile.split("_")
        full_path = os.path.join(logs, logfile)

        row_id = tokens[0]
        run_id = tokens[1]

        # print(full_path)
        data = read_csv_into_dict(row_id, full_path, exec_ref)
        energy, delta_t, intermediate_values = compute_average_energy(
            len(data["time"]), data["current"], data["time"], voltage=5)

        # write all intermediate values
        ref = get_reference_row(rows, row_id)
        for i in intermediate_values:
            writer_all.writerow([ref.depl, ref.model, i[1], i[2]])


        #print(f"row_id:{tokens[0]} run_id:{tokens[1]} - {energy} (joules)")
        measures[row_id] = {"run_id": run_id,
                               "eng": energy, "delta_t": delta_t}

        # check if total time matches exec time (log correlate)
        _, _, tmp = get_exec_time(exec_ref, row_id)
        exec_t = int(float(tmp))
        if abs(delta_t - exec_t) >= 5: 
            print("Deleting invalid data for row_id:", row_id, delta_t, exec_t)
            del measures[row_id]

    e_data = {}
    t_data = {}

    #print(measures)
    msr = OrderedDict(sorted(measures.items(), key=lambda t: int(t[0])))
    for i in msr.items():
        # print(i)
        rr = get_reference_row(rows, i[0])
        writer.writerow([i[0], i[1]["run_id"], rr.depl,
                        rr.model, i[1]["eng"], i[1]["delta_t"]])

        index = int(i[0]) % 12
        if index == 0:
            index = 12
        # print(index)

        if int(i[0]) <= 12:
            # print("lower")
            e_data[index] = [float(i[1]["eng"])]
            t_data[index] = [int(i[1]["delta_t"])]

        elif int(i[0]) >= 109:
            #print(index)
            e_data[index].append(float(i[1]["eng"]))
            t_data[index].append(int(i[1]["delta_t"]))

            row = [rr.depl, rr.model,
                   st.mean(e_data[index]), st.median(
                       e_data[index]), st.stdev(e_data[index]),
                   st.mean(t_data[index]), st.median(t_data[index]), st.stdev(t_data[index])]
            
            # print(row)
            # compute_iqr(e_data[index])
            # compute_iqr(t_data[index])
            writer_avg.writerow(row)
        else:
            # print("mid")
            e_data[index].append(float(i[1]["eng"]))
            t_data[index].append(int(i[1]["delta_t"]))

    csv_avg_out.close()
    csv_out.close()


write_eng(ref_file=REF_ROWS_RPI3, eng_file=RPI3_ENERGY_RESULTS,
          avg_file=RPI3_AVG, all_file=ALL_RPI3, logs=POWERLOGS_RPI3, exec_ref=EXEC_3)
write_eng(ref_file=REF_ROWS_RPI4, eng_file=RPI4_ENERGY_RESULTS,
          avg_file=RPI4_AVG, all_file=ALL_RPI4, logs=POWERLOGS_RPI4, exec_ref=EXEC_4)


#data = read_csv_into_dict("/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/powerlogs/26_3_measurements.csv")
#energy = compute_average_energy(len(data["time"]), data["current"], data["time"], voltage=5.2)
# print(data)
#print(energy, "(joules)")
