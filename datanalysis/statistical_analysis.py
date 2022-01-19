import os
import csv
import scipy.stats as stats
import scipy
import matplotlib
import numpy
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.stats.multitest as multitest
import pandas as pd
import seaborn as sns
import numpy as np
import cliffs_delta as cd
from multipy.fwer import holm_bonferroni

SUBJECTS = ['mnist', 'emotion', 'cifar10', 'yolov4']
CONTAINERS = ['ONNX', 'SCLB', 'DOCKER']
METRICS = ['energy']
DEVICES = ['rpi4', 'rpi3']

strategy_mapping = {0: 'bmetal', 1: 'wasm', 2: 'docker'}

# rpi4, rpi3
# onnx (bmetal), sclb (wasm), docker
all_energy = [[[], [], []], [[], [], []]]
all_exec = [[[], [], []], [[], [], []]]
all_mem = [[[], [], []], [[], [], []]]
all_mem_vsz = [[[], [], []], [[], [], []]]
all_cpu = [[[], [], []], [[], [], []]]
all_cpu_top = [[[], [], []], [[], [], []]]

all_energy_time = [[[], [], []], [[], [], []]]
all_mem_time = [[[], [], []], [[], [], []]]
all_mem_vsz_time = [[[], [], []], [[], [], []]]
all_cpu_time = [[[], [], []], [[], [], []]]
all_cpu_top_time = [[[], [], []], [[], [], []]]

print_vals = False

def read_energy(file_path):

    energy = ([], [], [])
    with open(file_path, "r") as csvf:
        for line in csvf.readlines()[1:]:
            tokens = line.split(",")
            if int(tokens[5]) > 200:
                continue

            if 'ONNX' in tokens:
                energy[0].append(int(float(tokens[4]) / 1000))
            elif 'SCLB' in tokens:
                energy[1].append(int(float(tokens[4]) / 1000))
            elif 'DOCKER' in tokens:
                energy[2].append(int(float(tokens[4]) / 1000))
            else:
                print("Invalid token for container type.")

    return energy

def read_all_energy(file_path):
    
    energy = ([], [], [])
    with open(file_path, "r") as csvf:
        for line in csvf.readlines()[1:]:
            tokens = line.split(",")

            if 'ONNX' in tokens:
                energy[0].append((int(float(tokens[2]) / 1000), float(tokens[3])))
            elif 'SCLB' in tokens:
                energy[1].append((int(float(tokens[2]) / 1000), float(tokens[3])))
            elif 'DOCKER' in tokens:
                energy[2].append((int(float(tokens[2]) / 1000), float(tokens[3])))
            else:
                print("Invalid token for container type.")

    return energy


def read_exec(file_path):
    
    data = ([], [], [])
    with open(file_path, "r") as csvf:
        for line in csvf.readlines()[1:]:
            tokens = line.split(",")

            if 'ONNX' in tokens:
                data[0].append(int(float(tokens[4])))
            elif 'SCLB' in tokens:
                data[1].append(int(float(tokens[4])))
            elif 'DOCKER' in tokens:
                data[2].append(int(float(tokens[4])))
            else:
                print("Invalid token for container type.")

    return data

def read_memcpu(file_path, type):
    
    data = ([], [], [])
    data_time = ([], [], [])

    with open(file_path, "r") as csvf:
        for line in csvf.readlines()[1:]:
            tokens = line.split(",")

            if type == 'cpu':
                val = float(tokens[2])
                timestamp = float(tokens[6])
            elif type == 'cpu_top':
                val = float(tokens[3])
                timestamp = float(tokens[7])
            elif type == 'mem':
                val = int(tokens[2]) / 1000 #mbytes
                timestamp = float(tokens[5])
                if val == 0:
                    continue
            elif type == 'vsz_mem':
                val = int(tokens[3]) / 1000 #mbytes
                timestamp = float(tokens[5])
                if val == 0:
                    continue
            else:
                print("Invalid type to parse")

            if 'ONNX' in tokens:
                data[0].append(val)
                data_time[0].append((val, timestamp))
            elif 'SCLB' in tokens:
                data[1].append(val)
                data_time[1].append((val, timestamp))
            elif 'DOCKER' in tokens:
                data[2].append(val)
                data_time[2].append((val, timestamp))
            else:
                print("Invalid token for container type.")

    return data, data_time

def kruscal_wallis(data):

    k_result = stats.kruskal(data[0], data[1], data[2])
    print(k_result, "p < 0.05:", k_result.pvalue < 0.05)
    return k_result.pvalue

def holm_correction(pvalues, times = 2):
    print("Initial pvalues: ", pvalues)
    aux = list(pvalues)
    
    for i in range(times):
        ret = multitest.multipletests(aux, alpha=0.05, method='holm')
        aux = ret[1]
        print(f"[{i}]", aux)

for i in range(len(DEVICES)):

    print(f'------------ {DEVICES[i]} ------------')

    # energy
    print(f'Starting analysis on [{DEVICES[i]}] - [total energy (kj)]')
    csv_file = f'./logs/energy_{DEVICES[i]}.csv'
    all_energy[i] = read_energy(csv_file)
    all_energy_time[i] = read_all_energy(f'./logs/all_energy_{DEVICES[i]}.csv')
    #print(all_energy_time)
    print('Data parsing complete')
    print('Analysis results')
    if print_vals:
        print('Min (onnx - sclb - docker): %s - %s - %s' %
            (str(min(all_energy[i][0])), str(min(all_energy[i][1])), str(min(all_energy[i][2]))))
        print('Max (onnx - sclb - docker): %s - %s - %s' %
            (str(max(all_energy[i][0])), str(max(all_energy[i][1])), str(max(all_energy[i][2]))))
        print('Median (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.median(all_energy[i][0])), str(numpy.median(all_energy[i][1])), str(numpy.median(all_energy[i][2]))))
        print('Mean (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.mean(all_energy[i][0])), str(numpy.mean(all_energy[i][1])), str(numpy.mean(all_energy[i][2]))))
        print('Std (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.std(all_energy[i][0])), str(numpy.std(all_energy[i][1])), str(numpy.std(all_energy[i][2]))))
        print('CV (onnx - sclb - docker): %s - %s - %s' %
            (str(scipy.stats.variation(all_energy[i][0])), str(scipy.stats.variation(all_energy[i][1])), str(scipy.stats.variation(all_energy[i][2]))))
        
        print('Shapiro-Wilk (onnx - sclb - docker): %s - %s - %s' %
            (str(stats.shapiro(all_energy[i][0])), str(stats.shapiro(all_energy[i][1])), str(stats.shapiro(all_energy[i][2]))))
    kw_energy = kruscal_wallis(all_energy[i])
    print(kw_energy)
    
    print('Cliffs delta (energy):')
    print('onnx - docker:', str(cd.cliffs_delta(all_energy[i][0], all_energy[i][2])))
    print('onnx - sclb:', str(cd.cliffs_delta(all_energy[i][0], all_energy[i][1])))
    print('docker - sclb', str(cd.cliffs_delta(all_energy[i][2], all_energy[i][1])))

    print(f'Starting analysis on [{DEVICES[i]}] - [exec time (s)]')
    csv_file = f'./logs/exec_{DEVICES[i]}.csv'
    all_exec[i] = read_exec(csv_file)
    #print(all_exec)
    print('Data parsing complete')
    print('Analysis results')
    if print_vals:
        print('Min (onnx - sclb - docker): %s - %s - %s' %
            (str(min(all_exec[i][0])), str(min(all_exec[i][1])), str(min(all_exec[i][2]))))
        print('Max (onnx - sclb - docker): %s - %s - %s' %
            (str(max(all_exec[i][0])), str(max(all_exec[i][1])), str(max(all_exec[i][2]))))
        print('Median (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.median(all_exec[i][0])), str(numpy.median(all_exec[i][1])), str(numpy.median(all_exec[i][2]))))
        print('Mean (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.mean(all_exec[i][0])), str(numpy.mean(all_exec[i][1])), str(numpy.mean(all_exec[i][2]))))
        print('Std (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.std(all_exec[i][0])), str(numpy.std(all_exec[i][1])), str(numpy.std(all_exec[i][2]))))
        print('CV (onnx - sclb - docker): %s - %s - %s' %
            (str(scipy.stats.variation(all_exec[i][0])), str(scipy.stats.variation(all_exec[i][1])), str(scipy.stats.variation(all_exec[i][2]))))
 
        print('Shapiro-Wilk (onnx - sclb - docker): %s - %s - %s' %
            (str(stats.shapiro(all_exec[i][0])), str(stats.shapiro(all_exec[i][1])), str(stats.shapiro(all_exec[i][2]))))
    kw_exec = kruscal_wallis(all_exec[i])
    print(kw_exec)
    
    print('Cliffs delta (execution):')
    print('onnx - docker:', str(cd.cliffs_delta(all_exec[i][0], all_exec[i][2])))
    print('onnx - sclb:', str(cd.cliffs_delta(all_exec[i][0], all_exec[i][1])))
    print('docker - sclb', str(cd.cliffs_delta(all_exec[i][2], all_exec[i][1])))

    print(f'Starting analysis on [{DEVICES[i]}] - [resident set size (mb)]')
    csv_file = f'./logs/all_mem_{DEVICES[i]}.csv'
    all_mem[i], all_mem_time[i] = read_memcpu(csv_file, 'mem')
    #print(all_exec)
    print('Data parsing complete')
    print('Analysis results')
    if print_vals:
        print('Min (onnx - sclb - docker): %s - %s - %s' %
            (str(min(all_mem[i][0])), str(min(all_mem[i][1])), str(min(all_mem[i][2]))))
        print('Max (onnx - sclb - docker): %s - %s - %s' %
            (str(max(all_mem[i][0])), str(max(all_mem[i][1])), str(max(all_mem[i][2]))))
        print('Median (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.median(all_mem[i][0])), str(numpy.median(all_mem[i][1])), str(numpy.median(all_mem[i][2]))))
        print('Mean (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.mean(all_mem[i][0])), str(numpy.mean(all_mem[i][1])), str(numpy.mean(all_mem[i][2]))))
        print('Std (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.std(all_mem[i][0])), str(numpy.std(all_mem[i][1])), str(numpy.std(all_mem[i][2]))))
        print('CV (onnx - sclb - docker): %s - %s - %s' %
            (str(scipy.stats.variation(all_mem[i][0])), str(scipy.stats.variation(all_mem[i][1])), str(scipy.stats.variation(all_mem[i][2]))))
        
        print('Shapiro-Wilk (onnx - sclb - docker): %s - %s - %s' %
            (str(stats.shapiro(all_mem[i][0])), str(stats.shapiro(all_mem[i][1])), str(stats.shapiro(all_mem[i][2]))))
    kw_rss = kruscal_wallis(all_mem[i])
    print(kw_rss)

    print('Cliffs delta (rss):')
    print('onnx - docker:', str(cd.cliffs_delta(all_mem[i][0], all_mem[i][2])))
    print('onnx - sclb:', str(cd.cliffs_delta(all_mem[i][0], all_mem[i][1])))
    print('docker - sclb', str(cd.cliffs_delta(all_mem[i][2], all_mem[i][1])))

    print(f'Starting analysis on [{DEVICES[i]}] - [virtual memory size (mb)]')
    csv_file = f'./logs/all_mem_{DEVICES[i]}.csv'
    all_mem_vsz[i], all_mem_vsz_time[i] = read_memcpu(csv_file, 'vsz_mem')
    #print(all_exec)
    print('Data parsing complete')
    print('Analysis results')
    if print_vals:
        print('Min (onnx - sclb - docker): %s - %s - %s' %
            (str(min(all_mem_vsz[i][0])), str(min(all_mem_vsz[i][1])), str(min(all_mem_vsz[i][2]))))
        print('Max (onnx - sclb - docker): %s - %s - %s' %
            (str(max(all_mem_vsz[i][0])), str(max(all_mem_vsz[i][1])), str(max(all_mem_vsz[i][2]))))
        print('Median (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.median(all_mem_vsz[i][0])), str(numpy.median(all_mem_vsz[i][1])), str(numpy.median(all_mem_vsz[i][2]))))
        print('Mean (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.mean(all_mem_vsz[i][0])), str(numpy.mean(all_mem_vsz[i][1])), str(numpy.mean(all_mem_vsz[i][2]))))
        print('Std (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.std(all_mem_vsz[i][0])), str(numpy.std(all_mem_vsz[i][1])), str(numpy.std(all_mem_vsz[i][2]))))
        print('CV (onnx - sclb - docker): %s - %s - %s' %
            (str(scipy.stats.variation(all_mem_vsz[i][0])), str(scipy.stats.variation(all_mem_vsz[i][1])), str(scipy.stats.variation(all_mem_vsz[i][2]))))
 
        print('Shapiro-Wilk (onnx - sclb - docker): %s - %s - %s' %
            (str(stats.shapiro(all_mem_vsz[i][0])), str(stats.shapiro(all_mem_vsz[i][1])), str(stats.shapiro(all_mem_vsz[i][2]))))
    kw_vsz = kruscal_wallis(all_mem_vsz[i])
    print(kw_vsz)

    print('Cliffs delta (vsz):')
    print('onnx - docker:', str(cd.cliffs_delta(all_mem_vsz[i][0], all_mem_vsz[i][2])))
    print('onnx - sclb:', str(cd.cliffs_delta(all_mem_vsz[i][0], all_mem_vsz[i][1])))
    print('docker - sclb', str(cd.cliffs_delta(all_mem_vsz[i][2], all_mem_vsz[i][1])))

    print(f'Starting analysis on [{DEVICES[i]}] - [cpu (%)]')
    csv_file = f'./logs/all_cpu_{DEVICES[i]}.csv'
    all_cpu[i], all_cpu_time[i] = read_memcpu(csv_file, 'cpu')
    #print(all_exec)
    print('Data parsing complete')
    print('Analysis results')
    if print_vals:
        print('Min (onnx - sclb - docker): %s - %s - %s' %
            (str(min(all_cpu[i][0])), str(min(all_cpu[i][1])), str(min(all_cpu[i][2]))))
        print('Max (onnx - sclb - docker): %s - %s - %s' %
            (str(max(all_cpu[i][0])), str(max(all_cpu[i][1])), str(max(all_cpu[i][2]))))
        print('Median (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.median(all_cpu[i][0])), str(numpy.median(all_cpu[i][1])), str(numpy.median(all_cpu[i][2]))))
        print('Mean (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.mean(all_cpu[i][0])), str(numpy.mean(all_cpu[i][1])), str(numpy.mean(all_cpu[i][2]))))
        print('Std (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.std(all_cpu[i][0])), str(numpy.std(all_cpu[i][1])), str(numpy.std(all_cpu[i][2]))))
        print('CV (onnx - sclb - docker): %s - %s - %s' %
            (str(scipy.stats.variation(all_cpu[i][0])), str(scipy.stats.variation(all_cpu[i][1])), str(scipy.stats.variation(all_cpu[i][2]))))
 
        print('Shapiro-Wilk (onnx - sclb - docker): %s - %s - %s' %
            (str(stats.shapiro(all_cpu[i][0])), str(stats.shapiro(all_cpu[i][1])), str(stats.shapiro(all_cpu[i][2]))))
    kw_cpu = kruscal_wallis(all_cpu[i])
    print(kw_cpu)


    print('Cliffs delta (pcpu):')
    print('onnx - docker:', str(cd.cliffs_delta(all_cpu[i][0], all_cpu[i][2])))
    print('onnx - sclb:', str(cd.cliffs_delta(all_cpu[i][0], all_cpu[i][1])))
    print('docker - sclb', str(cd.cliffs_delta(all_cpu[i][2], all_cpu[i][1])))

    print(f'Starting analysis on [{DEVICES[i]}] - [cpu_top (%)]')
    csv_file = f'./logs/all_cpu_{DEVICES[i]}.csv'
    all_cpu_top[i], all_cpu_top_time[i] = read_memcpu(csv_file, 'cpu_top')
    #print(all_exec)
    print('Data parsing complete')
    print('Analysis results')
    if print_vals:
        print('Min (onnx - sclb - docker): %s - %s - %s' %
            (str(min(all_cpu_top[i][0])), str(min(all_cpu_top[i][1])), str(min(all_cpu_top[i][2]))))
        print('Max (onnx - sclb - docker): %s - %s - %s' %
            (str(max(all_cpu_top[i][0])), str(max(all_cpu_top[i][1])), str(max(all_cpu_top[i][2]))))
        print('Median (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.median(all_cpu_top[i][0])), str(numpy.median(all_cpu_top[i][1])), str(numpy.median(all_cpu_top[i][2]))))
        print('Mean (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.mean(all_cpu_top[i][0])), str(numpy.mean(all_cpu_top[i][1])), str(numpy.mean(all_cpu_top[i][2]))))
        print('Std (onnx - sclb - docker): %s - %s - %s' %
            (str(numpy.std(all_cpu_top[i][0])), str(numpy.std(all_cpu_top[i][1])), str(numpy.std(all_cpu_top[i][2]))))
        print('CV (onnx - sclb - docker): %s - %s - %s' %
            (str(scipy.stats.variation(all_cpu_top[i][0])), str(scipy.stats.variation(all_cpu_top[i][1])), str(scipy.stats.variation(all_cpu_top[i][2]))))
 
        print('Shapiro-Wilk (onnx - sclb - docker): %s - %s - %s' %
            (str(stats.shapiro(all_cpu_top[i][0])), str(stats.shapiro(all_cpu_top[i][1])), str(stats.shapiro(all_cpu_top[i][2]))))
    kw_tcpu = kruscal_wallis(all_cpu_top[i])
    print(kw_tcpu)


    print('Cliffs delta (tcpu):')
    print('onnx - docker:', str(cd.cliffs_delta(all_cpu_top[i][0], all_cpu_top[i][2])))
    print('onnx - sclb:', str(cd.cliffs_delta(all_cpu_top[i][0], all_cpu_top[i][1])))
    print('docker - sclb', str(cd.cliffs_delta(all_cpu_top[i][2], all_cpu_top[i][1])))

    print('Corrections')
    kw_pvalues = [kw_energy, kw_exec, kw_rss, kw_vsz, kw_cpu, kw_tcpu]
    holm_correction(kw_pvalues)

    print('Corrections - mem')
    kw_mem_pvalues = [kw_rss, kw_vsz]
    holm_correction(kw_mem_pvalues)

    print('Corrections - cpu')
    kw_cpu_pvalues = [kw_cpu, kw_tcpu]
    holm_correction(kw_cpu_pvalues)

def create_and_cleanup_dataframe(data):
    
    df = pd.DataFrame(list(zip(data[0], data[1], data[2])), columns =['bmetal', 'wasm', 'docker'])
    #print(df)
    return df

    # in case we plan on using any type of correction
    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_df = df[filtered_entries]
    return new_df

def create_scatter_dataframe(data):
    # for every dependent variable (except execution time):
    # value | time | depl 

    #print("create_scatter_dataframe", len(data))
    #print("create_scatter_dataframe", len(data[0]))

    values = []
    times = []
    depls = []

    for i in range(3):
        # i = 0: bmetal, i = 1: wasm, i = 2: docker
        for tp in data[i]:
            #print("TP:", tp)
            #return
            values.append(tp[0])
            times.append(tp[1])
            depls.append(strategy_mapping[i])

    columns = ['value', 'timestamp', 'treatment']
    data = list(zip(values, times, depls))
    df = pd.DataFrame(data=data, columns=columns)

    #print(df)
    return df

    markers = {"bmetal": ".", "wasm": ".", "docker": "."}
    sns.scatterplot(x="timestamp", y="value", hue="treatment", data=df, markers=markers, palette="Greens")
    plt.show()

    return df 

def draw_qq_plot():

    fig, axs = plt.subplots(2, 6, tight_layout=True)

    # rpi4
    energy = all_energy[0][0] + all_energy[0][1] + all_energy[0][2]
    scipy.stats.probplot(energy, dist="norm", plot=axs[0][0])
    axs[0][0].set_title('Energy Consumption')
    axs[0][0].set_xlabel('Normal Theoretical Quantile')
    axs[0][0].set_ylabel('Ordered Quantile (kJ)')

    execution = all_exec[0][0] + all_exec[0][1] + all_exec[0][2]
    scipy.stats.probplot(execution, dist="norm", plot=axs[0][1])
    axs[0][1].set_title('Execution Time')
    axs[0][1].set_xlabel('Normal Theoretical Quantile')
    axs[0][1].set_ylabel('Ordered Quantile (seconds)')

    cpu = all_cpu[0][0] + all_cpu[0][1] + all_cpu[0][2]
    scipy.stats.probplot(cpu, dist="norm", plot=axs[0][2])
    axs[0][2].set_title('CPU (pidstat)')
    axs[0][2].set_xlabel('Normal Theoretical Quantile')
    axs[0][2].set_ylabel('Ordered Quantile (%)')
    axs[0][2].set_ylim(([0, 100]))

    cpu_top = all_cpu_top[0][0] + all_cpu_top[0][1] + all_cpu_top[0][2]
    scipy.stats.probplot(cpu_top, dist="norm", plot=axs[0][3])
    axs[0][3].set_title('CPU (top)')
    axs[0][3].set_xlabel('Normal Theoretical Quantile')
    axs[0][3].set_ylabel('Ordered Quantile (%)')

    mem = all_mem[0][0] + all_mem[0][1] + all_mem[0][2]
    scipy.stats.probplot(mem, dist="norm", plot=axs[0][4])
    axs[0][4].set_title('Resident Set Size')
    axs[0][4].set_xlabel('Normal Theoretical Quantile')
    axs[0][4].set_ylabel('Ordered Quantile (Mb)')

    mem_vsz = all_mem_vsz[0][0] + all_mem_vsz[0][1] + all_mem_vsz[0][2]
    scipy.stats.probplot(mem_vsz, dist="norm", plot=axs[0][5])
    axs[0][5].set_title('Virtual Memory Size')
    axs[0][5].set_xlabel('Normal Theoretical Quantile')
    axs[0][5].set_ylabel('Ordered Quantile (Mb)')

    # rpi3
    energy = all_energy[1][0] + all_energy[1][1] + all_energy[1][2]
    scipy.stats.probplot(energy, dist="norm", plot=axs[1][0])
    axs[1][0].set_title('Energy Consumption')
    axs[1][0].set_xlabel('Normal Theoretical Quantile')
    axs[1][0].set_ylabel('Ordered Quantile (kJ)')

    execution = all_exec[1][0] + all_exec[1][1] + all_exec[1][2]
    scipy.stats.probplot(execution, dist="norm", plot=axs[1][1])
    axs[1][1].set_title('Execution Time')
    axs[1][1].set_xlabel('Normal Theoretical Quantile')
    axs[1][1].set_ylabel('Ordered Quantile (seconds)')

    cpu = all_cpu[1][0] + all_cpu[1][1] + all_cpu[1][2]
    scipy.stats.probplot(cpu, dist="norm", plot=axs[1][2])
    axs[1][2].set_title('CPU (pidstat)')
    axs[1][2].set_xlabel('Normal Theoretical Quantile')
    axs[1][2].set_ylabel('Ordered Quantile (%)')
    axs[1][2].set_ylim(([0, 100]))

    cpu_top = all_cpu_top[1][0] + all_cpu_top[1][1] + all_cpu_top[1][2]
    scipy.stats.probplot(cpu_top, dist="norm", plot=axs[1][3])
    axs[1][3].set_title('CPU (top)')
    axs[1][3].set_xlabel('Normal Theoretical Quantile')
    axs[1][3].set_ylabel('Ordered Quantile (%)')

    mem = all_mem[1][0] + all_mem[1][1] + all_mem[1][2]
    scipy.stats.probplot(mem, dist="norm", plot=axs[1][4])
    axs[1][4].set_title('Resident Set Size')
    axs[1][4].set_xlabel('Normal Theoretical Quantile')
    axs[1][4].set_ylabel('Ordered Quantile (Mb)')

    mem_vsz = all_mem_vsz[1][0] + all_mem_vsz[1][1] + all_mem_vsz[1][2]
    scipy.stats.probplot(mem_vsz, dist="norm", plot=axs[1][5])
    axs[1][5].set_title('Virtual Memory Size')
    axs[1][5].set_xlabel('Normal Theoretical Quantile')
    axs[1][5].set_ylabel('Ordered Quantile (Mb)')

    plt.show()

def draw_violin_plot():
    fig, axs = plt.subplots(2, 6, tight_layout=True)

    # rpi4
    data_to_plot = create_and_cleanup_dataframe(all_energy[0])
    axs[0][0] = sns.violinplot(data=data_to_plot, ax=axs[0][0], palette="Greens", linewidth=0.8, scale='count')
    axs[0][0].set_title('Energy Consumption')
    axs[0][0].set_ylabel('Energy Consumption (kJ)')

    data_to_plot = create_and_cleanup_dataframe(all_exec[0])
    #print(data_to_plot)
    axs[0][1] = sns.violinplot(data=data_to_plot, ax=axs[0][1], palette="Greens", linewidth=0.8, scale='count')
    axs[0][1].set_title('Execution time')
    axs[0][1].set_ylabel('Execution time (seconds)')

    data_to_plot = create_and_cleanup_dataframe(all_cpu[0])
    axs[0][2] = sns.violinplot(data=data_to_plot, ax=axs[0][2], palette="Greens", linewidth=0.8, scale='count')
    axs[0][2].set_title('CPU (pidstat)')
    axs[0][2].set_ylabel('CPU (%)')
    axs[0][2].set_ylim(([0, 100]))

    data_to_plot = create_and_cleanup_dataframe(all_cpu_top[0])
    axs[0][3] = sns.violinplot(data=data_to_plot, ax=axs[0][3], palette="Greens", linewidth=0.8, scale='count')
    axs[0][3].set_title('CPU (top)')
    axs[0][3].set_ylabel('CPU (%)')

    data_to_plot = create_and_cleanup_dataframe(all_mem[0])
    axs[0][4] = sns.violinplot(data=data_to_plot, ax=axs[0][4], palette="Greens", linewidth=0.8, scale='count')
    axs[0][4].set_title('Memory (RSS)')
    axs[0][4].set_ylabel('Memory (RSS) (Mb)')

    data_to_plot = create_and_cleanup_dataframe(all_mem_vsz[0])
    axs[0][5] = sns.violinplot(data=data_to_plot, ax=axs[0][5], palette="Greens", linewidth=0.8, scale='count')
    axs[0][5].set_title('Memory (VSZ)')
    axs[0][5].set_ylabel('Memory (VSZ) (Mb)')
    
    # rpi3 
    data_to_plot = create_and_cleanup_dataframe(all_energy[1])
    axs[1][0] = sns.violinplot(data=data_to_plot, ax=axs[1][0], palette="Blues", linewidth=0.8, scale='count')
    axs[1][0].set_title('Energy Consumption')
    axs[1][0].set_ylabel('Energy Consumption (kJ)')

    data_to_plot = create_and_cleanup_dataframe(all_exec[1])
    axs[1][1] = sns.violinplot(data=data_to_plot, ax=axs[1][1], palette="Blues", linewidth=0.8, scale='count')
    axs[1][1].set_title('Execution time')
    axs[1][1].set_ylabel('Execution time (seconds)')

    data_to_plot = create_and_cleanup_dataframe(all_cpu[1])
    axs[1][2] = sns.violinplot(data=data_to_plot, ax=axs[1][2], palette="Blues", linewidth=0.8, scale='count')
    axs[1][2].set_title('CPU (pidstat)')
    axs[1][2].set_ylabel('CPU (%)')
    axs[1][2].set_ylim(([0, 100]))

    data_to_plot = create_and_cleanup_dataframe(all_cpu_top[1])
    axs[1][3] = sns.violinplot(data=data_to_plot, ax=axs[1][3], palette="Blues", linewidth=0.8, scale='count')
    axs[1][3].set_title('CPU (top)')
    axs[1][3].set_ylabel('CPU (%)')

    data_to_plot = create_and_cleanup_dataframe(all_mem[1])
    axs[1][4] = sns.violinplot(data=data_to_plot, ax=axs[1][4], palette="Blues", linewidth=0.8, scale='count')
    axs[1][4].set_title('Memory (RSS)')
    axs[1][4].set_ylabel('Memory (RSS) (Mb)')

    data_to_plot = create_and_cleanup_dataframe(all_mem_vsz[1])
    axs[1][5] = sns.violinplot(data=data_to_plot, ax=axs[1][5], palette="Blues", linewidth=0.8, scale='count')
    axs[1][5].set_title('Memory (VSZ)')
    axs[1][5].set_ylabel('Memory (VSZ) (Mb)')

    plt.show()

def draw_scatter_plot():
    fig, axs = plt.subplots(2, 6, tight_layout=True)
    markers = {"bmetal": ".", "wasm": ".", "docker": "."}

    # rpi4
    data_to_plot = create_and_cleanup_dataframe(all_energy[0])
    axs[0][0] = sns.scatterplot(data=data_to_plot, ax=axs[0][0], palette="Greens", markers=markers)
    axs[0][0].set_title('Energy Consumption')
    axs[0][0].set_ylabel('Energy Consumption (kJ)')

    data_to_plot = create_and_cleanup_dataframe(all_exec[0])
    #print(data_to_plot)
    axs[0][1] = sns.scatterplot(data=data_to_plot, ax=axs[0][1], palette="Greens", markers=markers) 
    axs[0][1].set_title('Execution time')
    axs[0][1].set_ylabel('Execution time (seconds)')

    data_to_plot = create_and_cleanup_dataframe(all_cpu[0])
    axs[0][2] = sns.scatterplot(data=data_to_plot, ax=axs[0][2], palette="Greens", markers=markers)
    axs[0][2].set_title('CPU (pidstat)')
    axs[0][2].set_ylabel('CPU (%)')
    axs[0][2].set_ylim(([0, 100]))

    data_to_plot = create_and_cleanup_dataframe(all_cpu_top[0])
    axs[0][3] = sns.scatterplot(data=data_to_plot, ax=axs[0][3], palette="Greens", markers=markers)
    axs[0][3].set_title('CPU (top)')
    axs[0][3].set_ylabel('CPU (%)')

    data_to_plot = create_and_cleanup_dataframe(all_mem[0])
    axs[0][4] = sns.scatterplot(data=data_to_plot, ax=axs[0][4], palette="Greens", markers=markers)
    axs[0][4].set_title('Memory (RSS)')
    axs[0][4].set_ylabel('Memory (RSS) (Mb)')

    data_to_plot = create_and_cleanup_dataframe(all_mem_vsz[0])
    axs[0][5] = sns.scatterplot(data=data_to_plot, ax=axs[0][5], palette="Greens", markers=markers)
    axs[0][5].set_title('Memory (VSZ)')
    axs[0][5].set_ylabel('Memory (VSZ) (Mb)')
    
    # rpi3 
    data_to_plot = create_and_cleanup_dataframe(all_energy[1])
    axs[1][0] = sns.scatterplot(data=data_to_plot, ax=axs[1][0], palette="Blues", markers=markers)
    axs[1][0].set_title('Energy Consumption')
    axs[1][0].set_ylabel('Energy Consumption (kJ)')

    data_to_plot = create_and_cleanup_dataframe(all_exec[1])
    axs[1][1] = sns.scatterplot(data=data_to_plot, ax=axs[1][1], palette="Blues", markers=markers)
    axs[1][1].set_title('Execution time')
    axs[1][1].set_ylabel('Execution time (seconds)')

    data_to_plot = create_and_cleanup_dataframe(all_cpu[1])
    axs[1][2] = sns.scatterplot(data=data_to_plot, ax=axs[1][2], palette="Blues", markers=markers)
    axs[1][2].set_title('CPU (pidstat)')
    axs[1][2].set_ylabel('CPU (%)')
    axs[1][2].set_ylim(([0, 100]))

    data_to_plot = create_and_cleanup_dataframe(all_cpu_top[1])
    axs[1][3] = sns.scatterplot(data=data_to_plot, ax=axs[1][3], palette="Blues", markers=markers)
    axs[1][3].set_title('CPU (top)')
    axs[1][3].set_ylabel('CPU (%)')

    data_to_plot = create_and_cleanup_dataframe(all_mem[1])
    axs[1][4] = sns.scatterplot(data=data_to_plot, ax=axs[1][4], palette="Blues", markers=markers)
    axs[1][4].set_title('Memory (RSS)')
    axs[1][4].set_ylabel('Memory (RSS) (Mb)')

    data_to_plot = create_and_cleanup_dataframe(all_mem_vsz[1])
    axs[1][5] = sns.scatterplot(data=data_to_plot, ax=axs[1][5], palette="Blues", markers=markers)
    axs[1][5].set_title('Memory (VSZ)')
    axs[1][5].set_ylabel('Memory (VSZ) (Mb)')

    plt.show()

def draw_scatter_plot_v2():
    fig, axs = plt.subplots(2, 5, tight_layout=True)
    #markers = {"bmetal": ".", "wasm": ".", "docker": "."}
    markers = "o"

    # rpi4
    data_to_plot = create_scatter_dataframe(all_energy_time[0])
    axs[0][0] = sns.scatterplot(data=data_to_plot, x="timestamp", y="value", hue="treatment", ax=axs[0][0], marker=markers)
    axs[0][0].set_title('Energy Consumption')
    axs[0][0].set_ylabel('Energy Consumption (kJ)')
    axs[0][0].set_xlabel('Time(s)')

    data_to_plot = create_scatter_dataframe(all_cpu_time[0])
    axs[0][1] = sns.scatterplot(data=data_to_plot, x="timestamp", y="value", hue="treatment", ax=axs[0][1], marker=markers)
    axs[0][1].set_title('CPU (pidstat)')
    axs[0][1].set_ylabel('CPU (%)')
    axs[0][1].set_ylim(([0, 100]))
    axs[0][1].set_xlabel('Time(s)')

    data_to_plot = create_scatter_dataframe(all_cpu_top_time[0])
    axs[0][2] = sns.scatterplot(data=data_to_plot, x="timestamp", y="value", hue="treatment", ax=axs[0][2], marker=markers)
    axs[0][2].set_title('CPU (top)')
    axs[0][2].set_ylabel('CPU (%)')
    axs[0][2].set_xlabel('Time(s)')

    data_to_plot = create_scatter_dataframe(all_mem_time[0])
    axs[0][3] = sns.scatterplot(data=data_to_plot, x="timestamp", y="value", hue="treatment", ax=axs[0][3], marker=markers)
    axs[0][3].set_title('Memory (RSS)')
    axs[0][3].set_ylabel('Memory (RSS) (Mb)')
    axs[0][3].set_xlabel('Time(s)')

    data_to_plot = create_scatter_dataframe(all_mem_vsz_time[0])
    axs[0][4] = sns.scatterplot(data=data_to_plot, x="timestamp", y="value", hue="treatment", ax=axs[0][4], marker=markers)
    axs[0][4].set_title('Memory (VSZ)')
    axs[0][4].set_ylabel('Memory (VSZ) (Mb)')
    axs[0][4].set_xlabel('Time(s)')
    
    # rpi3 
    data_to_plot = create_scatter_dataframe(all_energy_time[1])
    axs[1][0] = sns.scatterplot(data=data_to_plot, x="timestamp", y="value", hue="treatment", ax=axs[1][0], marker=markers)
    axs[1][0].set_title('Energy Consumption')
    axs[1][0].set_ylabel('Energy Consumption (kJ)')
    axs[1][0].set_xlabel('Time(s)')

    data_to_plot = create_scatter_dataframe(all_cpu_time[1])
    axs[1][1] = sns.scatterplot(data=data_to_plot, x="timestamp", y="value", hue="treatment", ax=axs[1][1], marker=markers)
    axs[1][1].set_title('CPU (pidstat)')
    axs[1][1].set_ylabel('CPU (%)')
    axs[1][1].set_ylim(([0, 100]))
    axs[1][1].set_xlabel('Time(s)')

    data_to_plot = create_scatter_dataframe(all_cpu_top_time[1])
    axs[1][2] = sns.scatterplot(data=data_to_plot, x="timestamp", y="value", hue="treatment", ax=axs[1][2], marker=markers)
    axs[1][2].set_title('CPU (top)')
    axs[1][2].set_ylabel('CPU (%)')
    axs[1][2].set_xlabel('Time(s)')

    data_to_plot = create_scatter_dataframe(all_mem_time[1])
    axs[1][3] = sns.scatterplot(data=data_to_plot, x="timestamp", y="value", hue="treatment", ax=axs[1][3], marker=markers)
    axs[1][3].set_title('Memory (RSS)')
    axs[1][3].set_ylabel('Memory (RSS) (Mb)')
    axs[1][3].set_xlabel('Time(s)')

    data_to_plot = create_scatter_dataframe(all_mem_vsz_time[1])
    axs[1][4] = sns.scatterplot(data=data_to_plot, x="timestamp", y="value", hue="treatment", ax=axs[1][4], marker=markers)
    axs[1][4].set_title('Memory (VSZ)')
    axs[1][4].set_ylabel('Memory (VSZ) (Mb)')
    axs[1][4].set_xlabel('Time(s)')
    plt.show()

#draw_qq_plot()
#draw_violin_plot()
#draw_scatter_plot()
draw_scatter_plot_v2()

