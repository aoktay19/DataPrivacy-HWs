class TreeNode:
    def __init__(self,name, depth = 0, parent=None):
        self.name = name
        self.parent = parent
        self.depth = depth
        self.children = []

    def add_child(self, child_name, depth, parent):
        self.children.append(TreeNode(child_name, depth, parent))

    def get_child(self, name):
        for node in self.children:
            if node.name == name:
                return node
        return None

    def count_leaf_nodes(self) :
        if not self.children :
            # If the node has no children, it is a leaf node
            return 1
        else :
            # Recursively count leaf nodes in each child
            return sum(count_leaf_nodes(child) for child in self.children)


def construct_tree(file_path):
    root = None

    stack = []

    with open(file_path, 'r') as file:
        for line in file :
            # Calculate the depth of the current node
            depth = line.count('\t')
            # Remove leading tabs and newline characters
            clean_line = line.strip()
            if depth == 0:
                # Root node
                root = TreeNode(clean_line,depth)
                current_node = root
                parent_node = root
                stack.append(current_node)
            else :
                # Child node
                while len(stack) > depth:
                    stack.pop()
                print([node.name for node in stack])
                current_node = stack[-1]
                current_node.add_child(clean_line, depth, current_node)
                #print([node.name for node in current_node.children])
                stack.append(current_node.get_child(clean_line))
                #print(current_node.get_child(clean_line).name)
    return root

def print_tree(node, indent=0):
    print('\t' * indent + f"{node.name}" + f" {node.depth}" + (f" Parent: {node.parent.name}" if node.parent is not None else "") )
    for child in node.children:
        print_tree(child, indent + 1)

for i in range(0, 12, 3):
    print(i)

# Example usage:
file_path = "/Users/ali.oktay/Desktop/HW1/DGHs/marital-status.txt"
tree = construct_tree(file_path)
print_tree(tree)