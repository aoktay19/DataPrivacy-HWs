def find_equivalent_class_list(raw_dataset, k):
    return [raw_dataset[index:index + k] for index in range(0, len(raw_dataset), k)]


def dfs_in_dgh(root, node_name):
    if root is None:
        return None

    if root.name == node_name:
        return root

    stack, path = [root], set()

    while stack:
        vertex = stack.pop()
        if vertex is None or vertex in path:
            continue
        path.add(vertex)
        for neighbor in vertex.children:
            if neighbor.name == node_name:
                return neighbor
            stack.append(neighbor)
    return None

def traverse_generalize(node_list):
    depth_list = [node.depth for node in node_list]

    is_equal_depth = len(set(depth_list)) == 1

    if is_equal_depth:
        node_list_common = node_list
    else:
        minimum_depth = min(depth_list)
        node_list_common = find_nodes_common_depth(node_list, minimum_depth)

    return node_list_common

def generalize_over_nodes(node_list):
    while True:
        is_equal_domain = len(set(node.name for node in node_list)) == 1
        if is_equal_domain:
            return node_list
        else:
            node_list = [node.parent for node in node_list]

def find_nodes_common_depth(node_list, target_depth):
    node_list_traversed = list()

    for node in node_list:
        temp_node = node
        if node.depth != target_depth:
            while temp_node.depth != target_depth:
                temp_node = temp_node.parent
        node_list_traversed.append(temp_node)

    return node_list_traversed
