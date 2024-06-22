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

    def count_leaf_nodes(self):
        if not self.children:
            # If the node has no children, it is a leaf node
            return 1
        else:
            # Recursively count leaf nodes in each child
            return sum(child.count_leaf_nodes() for child in self.children)

    def lowest_common_ancestor(self, node2):
        if (self.depth == node2.depth):
            if (self.name == node2.name):
                return self
            else :
                return self.parent.lowest_common_ancestor(node2.parent)
        elif (self.depth > node2.depth):
            return self.parent.lowest_common_ancestor(node2)
        else :
            return self.lowest_common_ancestor(node2.parent)