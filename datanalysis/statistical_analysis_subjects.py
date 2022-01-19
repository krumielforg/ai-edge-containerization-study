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

SUBJECTS = ['mnist', 'emotion', 'cifar10', 'yolov4']
CONTAINERS = ['ONNX', 'SCLB', 'DOCKER']
DEVICES = ['rpi4', 'rpi3']

# onnx, sclb, docker
all_energy = [[[], [], []], [[], [], []]]
all_exec = [[[], [], []], [[], [], []]]
all_mem = [[[], [], []], [[], [], []]]
all_mem_vsz = [[[], [], []], [[], [], []]]
all_cpu = [[[], [], []], [[], [], []]]
all_cpu_top = [[[], [], []], [[], [], []]]

print_vals = False


def read_energy(file_path):

    energy = {}
    for c in CONTAINERS:
        energy[c] = {}
        for s in SUBJECTS:
            energy[c][s] = []

    with open(file_path, "r") as csvf:
        for line in csvf.readlines()[1:]:
            tokens = line.split(",")

            val = float(float(tokens[4]) / 1000)
            depl = tokens[2]
            model = tokens[3]
            energy[depl][model].append(val)
    return energy


def read_exec(file_path):

    data = {}
    for c in CONTAINERS:
        data[c] = {}
        for s in SUBJECTS:
            data[c][s] = []

    with open(file_path, "r") as csvf:
        for line in csvf.readlines()[1:]:
            tokens = line.split(",")

            val = int(float(tokens[4]))
            depl = tokens[2]
            model = tokens[3]
            data[depl][model].append(val)
    return data


def read_memcpu(file_path, type):

    data = {}
    for c in CONTAINERS:
        data[c] = {}
        for s in SUBJECTS:
            data[c][s] = []

    with open(file_path, "r") as csvf:
        for line in csvf.readlines()[1:]:
            tokens = line.split(",")

            if type == 'cpu':
                val = float(tokens[2])
            elif type == 'cpu_top':
                val = float(tokens[3])
            elif type == 'mem':
                val = int(tokens[2]) / 1000  # mbytes
                if val == 0:
                    continue
            elif type == 'vsz_mem':
                val = int(tokens[3]) / 1000  # mbytes
                if val == 0:
                    continue
            else:
                print("Invalid type to parse")

            depl = tokens[0]
            model = tokens[1]
            data[depl][model].append(val)
    return data


def create_and_cleanup_dataframe(data):

    df = pd.DataFrame(list(zip(data[SUBJECTS[0]], data[SUBJECTS[1]], data[SUBJECTS[2]], data[SUBJECTS[3]])),
                      columns=SUBJECTS)
    return df

    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_df = df[filtered_entries]
    #print (new_df)
    return new_df


for i in range(len(DEVICES)):

    print(f'------------ {DEVICES[i]} ------------')

    print(f'Starting analysis on [{DEVICES[i]}] - [total energy (kj)]')
    csv_file = f'./logs/energy_{DEVICES[i]}.csv'
    all_energy[i] = read_energy(csv_file)

    print(f'Starting analysis on [{DEVICES[i]}] - [exec time (s)]')
    csv_file = f'./logs/exec_{DEVICES[i]}.csv'
    all_exec[i] = read_exec(csv_file)

    print(f'Starting analysis on [{DEVICES[i]}] - [resident set size (Mb)]')
    csv_file = f'./logs/all_mem_{DEVICES[i]}.csv'
    all_mem[i] = read_memcpu(csv_file, 'mem')

    print(f'Starting analysis on [{DEVICES[i]}] - [virtual memory size (Mb)]')
    csv_file = f'./logs/all_mem_{DEVICES[i]}.csv'
    all_mem_vsz[i] = read_memcpu(csv_file, 'vsz_mem')

    print(f'Starting analysis on [{DEVICES[i]}] - [cpu (%)]')
    csv_file = f'./logs/all_cpu_{DEVICES[i]}.csv'
    all_cpu[i] = read_memcpu(csv_file, 'cpu')

    print(f'Starting analysis on [{DEVICES[i]}] - [cpu_top (%)]')
    csv_file = f'./logs/all_cpu_{DEVICES[i]}.csv'
    all_cpu_top[i] = read_memcpu(csv_file, 'cpu_top')


def draw_violin_plot(device):
    fig, axs = plt.subplots(3, 6, tight_layout=True)

    colors = ["Greens", "light:#5A9", "Greys"]

    # onnx
    for i in range(len(CONTAINERS)):


        data_to_plot = create_and_cleanup_dataframe(all_energy[device][CONTAINERS[i]])
        axs[i][0] = sns.violinplot(data=data_to_plot, ax=axs[i][0], palette=colors[i], linewidth=0.8, scale='count')
        axs[i][0].set_title('Energy Consumption')
        axs[i][0].set_ylabel('Energy Consumption (kJ)')

        data_to_plot = create_and_cleanup_dataframe(all_exec[device][CONTAINERS[i]])
        # print(data_to_plot)
        axs[i][1] = sns.violinplot(data=data_to_plot, ax=axs[i][1], palette=colors[i], linewidth=0.8, scale='count')
        axs[i][1].set_title('Execution time')
        axs[i][1].set_ylabel('Execution time (seconds)')

        data_to_plot = create_and_cleanup_dataframe(all_cpu[device][CONTAINERS[i]])
        axs[i][2] = sns.violinplot(data=data_to_plot, ax=axs[i][2], palette=colors[i], linewidth=0.8, scale='count')
        axs[i][2].set_title('CPU (pidstat)')
        axs[i][2].set_ylabel('CPU (%)')
        axs[i][2].set_ylim(([0, 100]))

        data_to_plot = create_and_cleanup_dataframe(all_cpu_top[device][CONTAINERS[i]])
        axs[i][3] = sns.violinplot(data=data_to_plot, ax=axs[i][3], palette=colors[i], linewidth=0.8, scale='count')
        axs[i][3].set_title('CPU (top)')
        axs[i][3].set_ylabel('CPU (%)')

        data_to_plot = create_and_cleanup_dataframe(all_mem[device][CONTAINERS[i]])
        axs[i][4] = sns.violinplot(data=data_to_plot, ax=axs[i][4], palette=colors[i], linewidth=0.8, scale='count')
        axs[i][4].set_title('Memory (RSS)')
        axs[i][4].set_ylabel('Memory (RSS) (Mb)')

        data_to_plot = create_and_cleanup_dataframe(all_mem_vsz[device][CONTAINERS[i]])
        axs[i][5] = sns.violinplot(data=data_to_plot, ax=axs[i][5], palette=colors[i], linewidth=0.8, scale='count')
        axs[i][5].set_title('Memory (VSZ)')
        axs[i][5].set_ylabel('Memory (VSZ) (Mb)')

    
    plt.show()


draw_violin_plot(0)
draw_violin_plot(1)
