from random_anonymizer import dfs_in_dgh

def calculate_dist_for_cluster(DGHs, row_domain_list, identifier_list, raw_dataset, index, row_index):
    dist = 0
    for domain in row_domain_list:
        if domain in identifier_list:
            dist += find_node_calculate_dist(DGHs[domain], raw_dataset[index][domain], raw_dataset[row_index][domain])

    return dist

def find_node_calculate_dist(root, first_node_name, second_node_name):
    dist1_node = dfs_in_dgh(root, first_node_name)
    dist2_node = dfs_in_dgh(root, second_node_name)
    return calculate_dist(dist1_node, dist2_node, root)

def calculate_dist(node1, node2, first):
    lca = node1.lowest_common_ancestor(node2)
    all_leaf = first.count_leaf_nodes()

    cost_LM1 = calculate_cost(node1, lca, all_leaf)
    cost_LM2 = calculate_cost(node2, lca, all_leaf)
    return cost_LM1 + cost_LM2

def calculate_cost(raw_nodes,lca, all_leaf):
    raw = (raw_nodes.count_leaf_nodes() - 1) / (all_leaf - 1)
    anon = (lca.count_leaf_nodes() - 1) / (all_leaf - 1)
    return abs(raw - anon)