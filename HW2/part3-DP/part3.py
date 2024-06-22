import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random
import csv

from numpy import sqrt, exp



''' Functions to implement '''

# TODO: Implement this function!
def read_dataset(file_path):
    result = pd.read_csv(file_path)
    return result


# TODO: Implement this function!
def get_histogram(dataset, state='TX', year='2020'):
    dataset['date'] = pd.to_datetime(dataset['date'])
    filtered_dataset_state = dataset.query("state == @state")
    annual_positives = filtered_dataset_state.loc[filtered_dataset_state['date'].dt.year == int(year)]
    monthly_totals = annual_positives['positive']

    # plt.bar(range(1, 13), monthly_totals.values)
    # plt.xlabel('Month')
    # plt.ylabel('Positive Numbers')
    # plt.title('Monthly Positive Numbers for ' + state + ' in ' + year)
    # plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    # plt.show()
    return list(monthly_totals.values)


# TODO: Implement this function!
def get_dp_histogram(dataset, state, year, epsilon, N):
    b = N / epsilon
    dataset['date'] = pd.to_datetime(dataset['date'])
    filtered_dataset_state = dataset.query("state == @state")
    annual_positives = filtered_dataset_state.loc[filtered_dataset_state['date'].dt.year == int(year)]

    list_annual_positive = list(annual_positives['positive'].values)
    for index in range(len(list_annual_positive)):
        list_annual_positive[index] += np.random.laplace(0, b)

    return list_annual_positive


# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    length = len(actual_hist)
    error = 0

    for index in range(len(actual_hist)):
        error += abs(actual_hist[index] - noisy_hist[index])

    return error / length


# TODO: Implement this function!
def epsilon_experiment(dataset, state, year, eps_values, N):

    actual_dataset = get_histogram(dataset, state, year)
    err_values = list()

    for epsilon in eps_values:
        noisy_dataset = get_dp_histogram(dataset, state, year, epsilon, N)
        error = calculate_average_error(actual_dataset, noisy_dataset)
        err_values.append(error)

    return err_values


# TODO: Implement this function!
def N_experiment(dataset, state, year, epsilon, N_values):
    actual_dataset = get_histogram(dataset, state, year)
    err_values = list()

    for N in N_values:
        noisy_dataset = get_dp_histogram(dataset, state, year, epsilon, N)
        error = calculate_average_error(actual_dataset, noisy_dataset)
        err_values.append(error)

    return err_values


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def max_deaths_exponential(dataset, state, year, epsilon):
    max_deaths = {}
    N = 1
    dataset['date'] = pd.to_datetime(dataset['date'])
    filtered_dataset_state = dataset.query("state == @state")
    annual_deaths = filtered_dataset_state.loc[filtered_dataset_state['date'].dt.year == int(year)]
    monthly_totals = annual_deaths['death']


    total = 0
    for month, values in zip(range(1,13), monthly_totals.values):
        max_deaths[month] = math.exp((epsilon * values) / (2 * N))
        total += max_deaths[month]
    for month in range(1,13):
        max_deaths[month] = max_deaths[month] / total

    return np.random.choice(list(max_deaths.keys()), 1, p=list(max_deaths.values()))


# TODO: Implement this function!
def exponential_experiment(dataset, state, year, epsilon_list):
    N = 1
    iteration_num = 10000
    dataset['date'] = pd.to_datetime(dataset['date'])
    filtered_dataset_state = dataset.query("state == @state")
    annual_deaths = filtered_dataset_state.loc[filtered_dataset_state['date'].dt.year == int(year)]
    monthly_totals = annual_deaths['death']

    monthly_totals_dict = dict(zip(range(1,13), monthly_totals.values))
    max_month_deaths = max(monthly_totals_dict, key=monthly_totals_dict.get)

    true_number_list = list()

    for epsilon in epsilon_list:
        true_number = 0
        for i in range(0, iteration_num):
            exponential_noise_death = max_deaths_exponential(dataset, state, year, epsilon)
            if exponential_noise_death[0] == max_month_deaths:
                true_number += 1
        true_number_list.append((true_number/iteration_num) * 100)

    return true_number_list



# FUNCTIONS TO IMPLEMENT END #


def main():
    filename = "covid19-states-history.csv"
    dataset = read_dataset(filename)
    
    state = "TX"
    year = "2020"


    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg = epsilon_experiment(dataset, state, year, eps_values, 2)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])


    print("**** N EXPERIMENT RESULTS ****")
    N_values = [1, 2, 4, 8]
    error_avg = N_experiment(dataset, state, year, 0.5, N_values)
    for i in range(len(N_values)):
        print("N = ", N_values[i], " error = ", error_avg[i])

    state = "WY"
    year = "2020"

    print("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 1.0]
    exponential_experiment_result = exponential_experiment(dataset, state, year, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])



if __name__ == "__main__":
    main()
