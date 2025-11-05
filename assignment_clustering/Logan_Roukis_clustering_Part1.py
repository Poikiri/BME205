import numpy as np
import matplotlib.pyplot as plt
import argparse

# K-Means clustering functions
def initialize_centroids(X, k):
    np.random.seed(42)
    return X[np.random.choice(X.shape[0], k, replace=False)]

def assign_clusters(X, centroids):
    # Assign each point to the nearest centroid
    return np.argmin(np.linalg.norm(X[:, None] - centroids, axis=2), axis=1)

def update_centroids(X, labels, k):
    # Update centroids as mean of assigned points
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

def has_converged(old, new, tol):
    # Check if centroids have shifted less than tol
    return np.all(np.linalg.norm(old - new, axis=1) < tol)

def kmeans(X, k=10, max_iters=300, tol=1e-4):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if has_converged(centroids, new_centroids, tol): 
            break
        centroids = new_centroids
    return labels, centroids

# Visualization
def save_centroids_as_image(centroids, filename):
    k = centroids.shape[0]
    fig, axes = plt.subplots(1, k, figsize=(k, 1))
    for i, ax in enumerate(axes):
        ax.imshow(centroids[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        ax.set_title(f'#{i}', fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Centroid images saved as {filename}")

# Clustering error calculation
def clustering_error(y, labels, k):
    err = 0
    for i in range(k):
        mask = labels == i
        if np.any(mask):
            majority = np.bincount(y[mask]).argmax()
            err += np.sum(y[mask] != majority)
    return err

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Means clustering on MNIST subset.")
    parser.add_argument("-k", type=int, required=True, help="Number of clusters.")
    args = parser.parse_args()
    k = args.k
    # Load data
    X = np.load("MNIST_X_subset.npy", allow_pickle=True)
    y = np.load("MNIST_y_subset.npy", allow_pickle=True)
    # Run K-Means
    labels, centroids = kmeans(X, k)
    # Save centroid images
    save_centroids_as_image(centroids, f"centroids_k{k}.png")
    # Print clustering error
    print(f"k={k}, ERROR={clustering_error(y, labels, k)}")
