import numpy as np
from matplotlib import pyplot as plt
from shapely import geometry, ops
import math
import random

""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

""" Helpers """


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result


def plot_grid(cell_percentages):
    max_lat = -8.58
    max_long = 41.18
    min_lat = -8.68
    min_long = 41.14

    background_image = plt.imread('porto.png')

    fig, ax = plt.subplots()
    ax.imshow(background_image, extent=[min_lat, max_lat, min_long, max_long], zorder=1)

    rec = [(min_lat, min_long), (min_lat, max_long), (max_lat, max_long), (max_lat, min_long)]
    nx, ny = 4, 5  # number of columns and rows  4,5

    polygon = geometry.Polygon(rec)
    minx, miny, maxx, maxy = polygon.bounds
    dx = (maxx - minx) / nx  # width of a small part
    dy = (maxy - miny) / ny  # height of a small part
    horizontal_splitters = [geometry.LineString([(minx, miny + i * dy), (maxx, miny + i * dy)]) for i in range(ny)]
    vertical_splitters = [geometry.LineString([(minx + i * dx, miny), (minx + i * dx, maxy)]) for i in range(nx)]
    splitters = horizontal_splitters + vertical_splitters

    result = polygon
    for splitter in splitters:
        result = geometry.MultiPolygon(ops.split(result, splitter))

    grids = list(result.geoms)

    for grid_index, grid in enumerate(grids):
        x, y = grid.exterior.xy
        ax.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

        count = cell_percentages[grid_index]
        count = round(count, 2)

        centroid = grid.centroid
        ax.annotate(str(count) + '%', (centroid.x, centroid.y), color='black', fontsize=12,
                    ha='center', va='center', zorder=3)

    plt.show()


# You can define your own helper functions here. #
def calculate_average_error(actual_hist, noisy_hist):
    length = len(actual_hist)
    error = 0

    for index in range(len(actual_hist)):
        error += abs(actual_hist[index] - noisy_hist[index])

    return error / length
### HELPERS END ###

""" Functions to implement """


# GRR

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    domain_size = len(DOMAIN)
    perturbation_prob = math.exp(epsilon) / (math.exp(epsilon) + domain_size - 1)
    perturbation_prob_other = (1 - perturbation_prob) / (domain_size - 1)

    prob_list = list(np.zeros(domain_size))
    for i in range(len(prob_list)):
        if i == val - 1:
            prob_list[i] = perturbation_prob
        else:
            prob_list[i] = perturbation_prob_other

    random_choice = random.choices(DOMAIN, prob_list, k=1)[0]
    return random_choice


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    N = len(perturbed_values)

    domain_size = len(DOMAIN)
    perturbation_prob = math.exp(epsilon) / (math.exp(epsilon) + domain_size - 1)
    perturbation_prob_other = (1 - perturbation_prob) / (domain_size - 1)

    I_v = np.zeros(domain_size)

    for val in perturbed_values:
        I_v[val - 1] += 1

    result = [(iv_val - (N * perturbation_prob_other)) / (perturbation_prob - perturbation_prob_other) for iv_val in I_v]

    return result


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    actual_val_list = [row for row in dataset]

    perturbed_val_list = [perturb_grr(row, epsilon) for row in actual_val_list]

    estimated_freq = estimate_grr(perturbed_val_list, epsilon)

    reports = np.zeros(len(estimated_freq))
    for val in actual_val_list:
        reports[val - 1] += 1

    number_of_reports = reports.tolist()
    return calculate_average_error(number_of_reports, estimated_freq)


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):
    d = len(DOMAIN)
    result = list(np.zeros(d))
    result[val - 1] = 1
    return result


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    perturbed_values = encoded_val.copy()
    perturbation_prob = math.exp(epsilon / 2) / (math.exp(epsilon / 2) + 1)

    for val in range(len(encoded_val)):
        random_number = random.uniform(0, 1)
        if random_number > perturbation_prob:
            perturbed_values[val] = int(encoded_val[val] == 0)

    return perturbed_values


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    N = len(perturbed_values)
    domain_size = len(DOMAIN)
    perturbation_prob = math.exp(epsilon/2) / (math.exp(epsilon/2) + 1)
    perturbation_prob_other = 1 - perturbation_prob

    perturbed_bit_vectors = np.array(perturbed_values)
    sum_of_perturbed_values = list(np.sum(perturbed_bit_vectors, axis=0))

    result = [(val - (N * perturbation_prob_other)) / (perturbation_prob - perturbation_prob_other) for val in sum_of_perturbed_values]

    return result


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    perturbed_values = list()
    domain_size = len(DOMAIN)
    real = np.zeros(domain_size)
    for row in dataset:
        encoded_values = encode_rappor(row)
        perturbed_values.append(perturb_rappor(encoded_values, epsilon))
        real += encoded_values
    return calculate_average_error(real.tolist(), estimate_rappor(perturbed_values, epsilon))


# OUE

# TODO: Implement this function!
def encode_oue(val):
    d = len(DOMAIN)
    result = list(np.zeros(d))
    result[val - 1] = 1
    return result


# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):
    perturbed_values = encoded_val.copy()
    perturbed_val_0_1 = 1 / (math.exp(epsilon) + 1)
    for val in range(len(encoded_val)):
        if encoded_val[val] == 0:
            rand_num = random.uniform(0, 1)
            if rand_num < perturbed_val_0_1:
                perturbed_values[val] = 1
        else:
            rand_num = random.randint(0, 1)
            perturbed_values[val] = rand_num
    return perturbed_values


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):
    n = len(perturbed_values)
    domain_size = len(DOMAIN)

    result = np.zeros(domain_size)

    for val in range(domain_size) :
        c_estimated = sum(1 for pert_val in perturbed_values if pert_val[val] == 1)
        c_real = (((((math.exp(epsilon)) + 1) * c_estimated) - n) * 2) / ((math.exp(epsilon)) - 1)
        result[val] = c_real

    return result.tolist()


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):
    domain_size = len(DOMAIN)
    perturbed_values = []
    actual_values = np.zeros(domain_size)

    for row in dataset:
        encoded = encode_oue(row)
        perturbed_values.append(perturb_oue(encoded, epsilon))
        actual_values += encoded

    estimated_freq = estimate_oue(perturbed_values, epsilon)
    plot_grid(estimated_freq)

    return calculate_average_error(actual_values, estimated_freq)


def main():
    dataset = read_dataset("taxi-locations.dat")
    """
    for epsilon in [0.01, 0.1, 0.5, 1, 2] :
        plot_grid([oue_experiment(dataset, epsilon) for _ in range(20)])"""
    print("GRR EXPERIMENT")
    for epsilon in [0.01, 0.1, 0.5, 1, 2]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.3f}".format(epsilon, error))

    print("*" * 50)

    print("RAPPOR EXPERIMENT")
    for epsilon in [0.01, 0.1, 0.5, 1, 2]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.3f}".format(epsilon, error))

    print("*" * 50)

    print("OUE EXPERIMENT")
    for epsilon in [0.01, 0.1, 0.5, 1, 2]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.3f}".format(epsilon, error))


            
    

if __name__ == "__main__":
    main()
