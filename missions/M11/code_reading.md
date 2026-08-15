# Code reading

Read the notebook's tree-inspection helpers before executing them.

For `describe_nodes`, identify:

1. which parallel `tree_` arrays describe a node;
2. how an internal node is distinguished from a leaf;
3. why the left branch means `feature <= threshold`;
4. where sample counts, impurity, and class counts come from;
5. how the predicted class is derived at a leaf.

For `explain_path`, identify:

1. the input row and feature order;
2. the node IDs returned by `decision_path`;
3. the comparison made at each internal node;
4. the first branch whose direction would change after a perturbation;
5. the terminal leaf and its training class distribution.

Before running either helper, manually trace one row through the textual tree exported by `export_text`. Record the expected node sequence and predicted class. After execution, compare the manual trace with the observed path and identify the first divergence, if any.

The reading is incomplete if it only paraphrases function names. Evidence must connect array values and comparisons to an actual row.
