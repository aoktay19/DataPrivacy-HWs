# from random_anonymizer import dfs_in_dgh
#
# def search_axe(dataframe, ID='income') :
#     """
#     Returns the axis according to which the cut will be made
#
#     Parameters:
#     ----------
#         dataframe : Partition P stored in a pandas dataframe
#         ID: Identifying column
#     """
#
#     widest_val = 0
#     axe = ""
#     columns = list(dataframe.columns)
#     columns.remove(str(ID))
#
#     for column in columns :
#         width = len(dataframe[column].unique())
#         if widest_val < width :
#             widest_val = width
#             axe = column
#
#     return axe
#
#
# def median_and_split(dataframe, axe, k, axe_tab, DGHs) :
#     """
#     Compute median on dataframe for given axis and make the split then return
#     the result
#
#     Parameters:
#     ----------
#         dataframe: Partition P stored in a pandas dataframe
#         axe: Axis to split dataset
#         k: the k for mondrian algorithm
#     """
#
#     axe_tab.append(axe)
#     nb_rows = len(dataframe[axe].unique())
#     rootNode = dfs_in_dgh(DGHs[axe], axe)
#     df_left = dataframe[(dataframe[axe] <= median)]
#     df_right = dataframe[(dataframe[axe] > median)]
#
#     """
#     childrenları al
#     split yap her elemanın valuesunu
#
#     """
#
#     if not is_a_good_split(df_left, df_right, k) :
#         # Moitié de chaque côté
#         df_left = dataframe.iloc[:k]
#         right_ix = -(len(dataframe) - k)
#         df_right = dataframe.iloc[right_ix :]
#
#     return df_left, df_right
#
#
# def is_splittable(dataset, k) :
#     """
#     Is this partition (dataset) splittable given this k?
#     Parameters:
#     ----------
#         dataset: A partition
#         k: the k for mondrian algorithm
#     """
#     return len(dataset) >= (2 * k)
#
#
# def is_a_good_split(df_left, df_right, k) :
#     """
#     Is this a good split given this value of k?
#
#     Parameters:
#     ----------
#         df_left & df_right: Partitions made from a bigger one
#         k: the k for mondrian algorithm
#     """
#     # No df empty and each one with at least k rows
#     return (len(df_left) >= k and len(df_right) >= k)
#
#
# def mondrian_anonymize(dataset, k, df_result, axe_tab, DGHs, ID='income') :
#     """
#     Compute mondrian algorithm on dataset with the given k
#
#     Parameters:
#     ----------
#         dataset: Initial dataset
#         k: the k for mondrian algorithm
#     """
#
#     if (not is_splittable(dataset, k)) :
#         print(len(dataset["age"]))
#         df_result.append(dataset)  # Can't split anymore
#     else :
#         axe = search_axe(dataset, ID)  # Choose axis
#         L_dataset, R_dataset = median_and_split(dataset, axe, k, axe_tab, DGHs)  # Compute median and split
#
#         # recursion for new partitions
#         mondrian_anonymize(L_dataset, k, df_result, axe_tab, DGHs, ID)
#         mondrian_anonymize(R_dataset, k, df_result, axe_tab, DGHs, ID)
#
#
# def reconstitution(df_result, columns, ID) :
#     """
#     Create dataframe from the array of all partitions created by mondian anonymize
#
#     Parameters:
#     ----------
#         df_result: array containing partitions as dataframes
#     """
#
#     mondrian_dataframe = []
#     columns_and_values = {}
#     for i in range(0, len(df_result)) :
#         columns_and_values = {}
#
#         for j in range(0, len(columns)) :
#
#             column = columns[j]
#             #print(column == ID)
#
#             if column != ID :
#                 column_interval = "[" + str(min(df_result[i][column])) + "," + str(max(df_result[i][column])) + "]"
#                 columns_and_values[column] = column_interval
#
#         for value in df_result[i][ID] :
#             row = columns_and_values
#             row[ID] = value
#             mondrian_dataframe.append(row)
#     return mondrian_dataframe

# Bir tane k anonomity chek

#bir tane splitter
#DGH' ın her key için child sayısına bakcak en küçük olanı alcak en küçük eşit varsa a bakcak sonra dataseti bölcekolşan listlerini

def calculate_L1_distance(specialization):
    length = len(specialization)

    sum = 0
    uniform_dist_elem = 1 / length
    for index in specialization:
        sum += abs(index - uniform_dist_elem)

    return sum

def is_splittable(dataset, k) :

    return len(dataset) >=  2 * k

def has_descendant_with_name(node, target_name) :
    if node.name == target_name :
        return True

    for child in node.children :
        if has_descendant_with_name(child, target_name) :
            return True

    return False

def dataset_splitter(dataset, DGHs, k):

    child_node_dict = {}
    for name, node in DGHs.items():
        child_node_dict[len(node.children)] = name

    #for node in child_node_dict[min(child_node_dict.keys())]
    child_node_dict[min(child_node_dict.keys())]





