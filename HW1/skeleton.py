##############################################################################
# This skeleton was created by Efehan Guner (efehanguner21@ku.edu.tr)    #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
import glob
import os
import sys
from copy import deepcopy
from typing import Optional
import numpy as np
import pandas as pd

from TreeNode import TreeNode
from random_anonymizer import *
from clustering import *
from top_down import *

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    # print(result[0]['age']) # debug: testing.
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True



def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    #TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.

    root = None
    stack = []

    with open(DGH_file, 'r') as file:
        for line in file :
            # Calculate the depth of the current node
            depth = line.count('\t')
            # Remove leading tabs and newline characters
            clean_line = line.strip()
            if depth == 0:
                # Root node
                root = TreeNode(clean_line,depth)
                current_node = root
                stack.append(current_node)
            else :
                # Child node
                while len(stack) > depth:
                    stack.pop()

                current_node = stack[-1]
                current_node.add_child(clean_line, depth, current_node)
                stack.append(current_node.get_child(clean_line))
    return root


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file);

    return DGHs

def get_attribute_depth(dgh, attribute_name):
    return dfs_in_dgh(dgh, attribute_name).depth
##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)
    #TODO: complete this function.

    identifier_list = list(DGHs.keys())
    cost_MD = 0
    for row in range(len(raw_dataset)):
        row_identifier_list = list(raw_dataset[row].keys())
        for identifier in row_identifier_list:
            if identifier in identifier_list:
                raw_data_name = raw_dataset[row][identifier]
                anonymized_data_name = anonymized_dataset[row][identifier]
                raw_data_depth = get_attribute_depth(DGHs[identifier], raw_data_name)
                anonymized_data_depth = get_attribute_depth(DGHs[identifier], anonymized_data_name)
                cost_MD += abs(raw_data_depth - anonymized_data_depth)

    return cost_MD


def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.

    identifier_list = list(DGHs.keys())
    cost_lm = 0
    weight = len(identifier_list)
    for row_index in range(len(raw_dataset)):
        for identifier in DGHs.keys():
            identifier_name = anonymized_dataset[row_index][identifier]
            lm_value = (dfs_in_dgh(DGHs[identifier], identifier_name).count_leaf_nodes() - 1) / (DGHs[identifier].count_leaf_nodes() - 1)
            cost_lm += lm_value * 1/weight

    return cost_lm


def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str, s: int):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)    

    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s) ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)
    
    #TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.
    # Store your results in the list named "clusters". 
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...
    clusters = find_equivalent_class_list(raw_dataset, k)

    for identifier_name, dgh_root in DGHs.items():
        identifier_name_dict = {}

        for equivalent_classes in clusters:
            list_of_nodes = []
            for ec in equivalent_classes:
                identifier_value = ec[identifier_name]

                if identifier_value in identifier_name_dict:
                    node = identifier_name_dict[identifier_value]
                else:
                    node = dfs_in_dgh(dgh_root, identifier_value)
                    identifier_name_dict[identifier_value] = node
                list_of_nodes.append(node)
            node_list = traverse_generalize(list_of_nodes)
            generalized_identifier_list = generalize_over_nodes(node_list)

            for data, node in zip(equivalent_classes, generalized_identifier_list):
                data[identifier_name] = node.name

    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters: #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)


def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []
    #TODO: complete this function.

    identifier_list = list(DGHs.keys())
    unused = [0] * len(raw_dataset)
    index = 0

    while unused.count(0) >= k:
        dist_list = []
        cluster = []
        if unused[index] == 0:
            unused[index] = 1
            rec = raw_dataset[index]
            cluster.append(rec)

            row_domain_list = list(rec.keys())

            for row_index in range(index+1, len(raw_dataset)):
                if unused[row_index] == 0:
                    dist = calculate_dist_for_cluster(DGHs, row_domain_list, identifier_list, raw_dataset, index, row_index)
                    dist_list.append((dist, row_index))

            dist_list = sorted(dist_list)[:k - 1]

            for dist_index in dist_list:
                unused[dist_index[1]] = 1
                cluster.append(raw_dataset[dist_index[1]])

            anonymized_dataset.append(cluster)

        index += 1

    last_cluster = []
    for i in range(len(unused)):
        if unused[i] == 0:
            last_cluster.append(raw_dataset[i])
    if len(last_cluster) != 0:
        anonymized_dataset.append(last_cluster)

    for identifier_name, dgh_root in DGHs.items():
        identifier_name_dict = {}
        for equivalent_classes in anonymized_dataset:
            list_of_nodes = []
            for ec in equivalent_classes:
                identifier_value = ec[identifier_name]
                if identifier_value in identifier_name_dict:
                    node = identifier_name_dict[identifier_value]
                else:
                    node = dfs_in_dgh(dgh_root, identifier_value)
                    identifier_name_dict[identifier_value] = node
                list_of_nodes.append(node)
            node_list = traverse_generalize(list_of_nodes)
            generalized_identifier_list = generalize_over_nodes(node_list)

            for data, node in zip(equivalent_classes, generalized_identifier_list):
                data[identifier_name] = node.name

    write_dataset(raw_dataset, output_file)


def topdown_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Top-down anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []
    #TODO: complete this function.
    axe_tab = []
    result_dict = {}

    # Iterate over the list of dictionaries
    for entry in raw_dataset :
        for key, value in entry.items() :
            # If the key is not already in the result_dict, add it with an empty list
            if key not in result_dict :
                result_dict[key] = []
            # Append the value to the list associated with the key
            result_dict[key].append(value)
    dataFrame = pd.DataFrame(result_dict)
    #mondrian_anonymize(dataFrame, k, anonymized_dataset, axe_tab, DGHs)
    #res = reconstitution(anonymized_dataset, list(result_dict.keys()), "income")
    write_dataset(anonymized_dataset, output_file)


#Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k")
    print(f"\tWhere algorithm is one of [clustering, random, topdown]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'topdown']:
    print("Invalid algorithm.")
    sys.exit(2)

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer")
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, topdown]")
        sys.exit(1)

    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

def print_tree(node, indent=0):
    for key, value in node.items():
        print('\t' * indent + key)
        print_tree(value, indent + 1)


# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300