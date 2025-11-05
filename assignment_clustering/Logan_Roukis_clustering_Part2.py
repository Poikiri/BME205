import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import collections

# Step 1: Load the Dogs dataset
# dogs_X.npy: SNP data (1355 samples x 784 features)
# dogs_clades.npy: Clade info (1355 samples)
dogs_X = np.load('dogs_X.npy')  # SNP feature matrix
dogs_clades = np.load('dogs_clades.npy', allow_pickle=True)  # True clade labels

# Step 2: Perform hierarchical clustering (Ward linkage)
Z = linkage(dogs_X, method='ward')

# Step 3: Plot truncated dendrogram with 30 leaf nodes
plt.figure(figsize=(12, 8))
k = 30
cluster_assignments = fcluster(Z, k, criterion='maxclust')
cluster_counts = collections.Counter(cluster_assignments)
leaf_labels = []
for leaf_cluster in np.unique(cluster_assignments):
    leaf_labels.append(str(cluster_counts[leaf_cluster]))

trunc_dendro = dendrogram(
    Z,
    truncate_mode='lastp',
    p=k,
    show_leaf_counts=True,  # Show sample count per cluster
    leaf_rotation=90,
    leaf_font_size=12,
    color_threshold=None
)

plt.title('Dogs Dendrogram (Ward-linkage, Truncated to 30 Clusters)')
plt.xlabel('Cluster (Number of Samples)')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('Dogs_dendrogram_truncated.png')
plt.close()

# Step 4: Compute clustering error
error = 0
for cluster_id in np.unique(cluster_assignments):
    indices = np.where(cluster_assignments == cluster_id)[0]
    clade_labels = dogs_clades[indices]
    majority_clade = collections.Counter(clade_labels).most_common(1)[0][0]
    cluster_error = np.sum(clade_labels != majority_clade)
    error += cluster_error

# Step 5: Print error in required format
print(f"k={k}, ERROR={error}")
