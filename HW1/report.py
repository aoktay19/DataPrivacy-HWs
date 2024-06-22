import time
import sys

import numpy as np

from skeleton import *

# importing the required module
import matplotlib.pyplot as plt


def k_vs_time(algo_type, raw_dataset_file, DGH_folder):
    list_of_k = [4, 8, 16, 32, 64, 128]
    list_of_seed = [5,10,15,20,25]
    list_of_time_result = list()
    list_of_average_time_result = list()
    #raw_dataset = read_dataset(raw_dataset_file)
    #DGHS = read_DGHs(DGH_folder)
    ## Time vs K
    for k in list_of_k:
        if algo_type == "random":
            for seed in list_of_seed:
                start = time.time()
                random_anonymizer(raw_dataset_file, DGH_folder, k, "output.csv", seed)
                end = time.time()
                list_of_time_result.append(end - start)
            average = sum(list_of_time_result) / len(list_of_time_result)
            list_of_average_time_result.append(average)
        elif algo_type == "clustering":
            for seed in list_of_seed:
                start = time.time()
                #raw_dataset = np.array(raw_dataset)
                np.random.seed(seed)  ## to ensure consistency between runs
                #np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize
                clustering_anonymizer(raw_dataset_file, DGH_folder, k, "output.csv")
                end = time.time()
                list_of_time_result.append(end - start)
            average = sum(list_of_time_result) / len(list_of_time_result)
            list_of_average_time_result.append(average)

    # plotting the points
    plt.plot(list_of_k, list_of_average_time_result)

    # naming the x axis
    plt.xlabel('K')
    # naming the y axis
    plt.ylabel('Time (second)')

    # giving a title to my graph
    plt.title('K vs Time')

    # function to show the plot
    plt.show()


def k_vs_cost(algo_type,raw_dataset_file, DGH_folder, cost_type):
    list_of_k = [4, 8, 16, 32, 64, 128]
    list_of_seed = [5, 10, 15, 20, 25]
    list_of_average_cost_result = list()
    name = ""
    #DGHs = read_DGHs(DGH_folder)
    ## Time vs K
    for k in list_of_k:
        list_of_cost_result = list()
        if algo_type == "random":
            for seed in list_of_seed:
                random_anonymizer(raw_dataset_file, DGH_folder, k, "output.csv", seed)
                cost = 0
                if cost_type == "LM":
                    name = "Random LM vs K"
                    cost = cost_LM(raw_dataset_file, "output.csv", DGH_folder)
                elif cost_type == "MD":
                    name = "Random MD vs K"
                    cost = cost_MD(raw_dataset_file, "output.csv", DGH_folder)
                list_of_cost_result.append(cost)
            average = sum(list_of_cost_result)
            list_of_average_cost_result.append(average)
        elif algo_type == "clustering":
            for seed in list_of_seed:
                #raw_dataset = np.array(raw_dataset)
                np.random.seed(seed)  ## to ensure consistency between runs
                #np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize
                clustering_anonymizer(raw_dataset_file, DGH_folder, k, "output.csv")
                cost = 0
                if cost_type == "LM":
                    name = "Clustering LM vs K"
                    cost = cost_LM(raw_dataset_file, "output.csv", DGH_folder)
                elif cost_type == "MD":
                    name = "Clustering MD vs K"
                    cost = cost_MD(raw_dataset_file, "output.csv", DGH_folder)
                list_of_cost_result.append(cost)
            average = sum(list_of_cost_result)
            list_of_average_cost_result.append(average)

    # plotting the points
    plt.plot(list_of_k, list_of_average_cost_result)

    # naming the x axis
    plt.xlabel('K')
    # naming the y axis
    plt.ylim([min(list_of_average_cost_result), max(list_of_average_cost_result)])
    plt.ylabel(cost_type)

    # giving a title to my graph
    plt.title(name)

    # function to show the plot
    plt.show()


algorithm = sys.argv[1]


dgh_path = sys.argv[2]
raw_file = sys.argv[3]
secondAlgo = sys.argv[4]
cost_type = sys.argv[5]

function = eval(f"k_vs_{algorithm}")
if function == k_vs_cost :
    function(secondAlgo, raw_file, dgh_path, cost_type)
else :
    function(secondAlgo, raw_file, dgh_path)