import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    np.set_printoptions(precision=4, suppress=True)

    # 1) Load data
    X = np.load('part1_data.npy', allow_pickle=True)
    n = X.shape[0]

    # 2) sklearn PCA (n_components=2)
    pca = PCA(n_components=2)
    pca.fit(X)
    x_pca = pca.transform(X)  # shape (n, 2)

    print("Sklearn direction of each PC (PC1, PC2):\n", pca.components_)
    print("Sklearn explained variance for each PC (PC1, PC2):\n", pca.explained_variance_)
    print("Sklearn fraction of variance explained by each PC (PC1, PC2):\n", pca.explained_variance_ratio_)
    print("Sklearn data projection into 1-D (onto PC1):\n", x_pca[:, 0])

    # 3) Center data
    mu = X.mean(axis=0)
    Xc = X - mu

    # 4) Covariance matrix (2x2)
    sigma = (Xc.T @ Xc) / (n - 1)
    print("\nCovariance matrix:\n", sigma)

    # 5) Eigendecomposition (ascending eigenvalues from eigh)
    evals, evecs = np.linalg.eigh(sigma)  # columns in evecs are eigenvectors
    order = np.argsort(evals)[::-1]       # descending order
    evals = evals[order]
    evecs = evecs[:, order]

    lam1, lam2 = float(evals[0]), float(evals[1])
    v1 = evecs[:, 0]
    v2 = evecs[:, 1]

    print(f"\nlambda_1 = {lam1:.6f}")
    print("v_1 = ", v1)
    print(f"lambda_2 = {lam2:.6f}")
    print("v_2 = ", v2)

    # Optional note: eigenvector sign is arbitrary. To align signs with sklearn, uncomment below:
    # if np.dot(v1, pca.components_[0]) < 0: v1 = -v1

    # Question 1: Compare manual eigendecomposition to sklearn outputs
    print("Question 1: The PCA components align with the covariance eigenvectors (up to sign), "
          "and the explained variances match the eigenvalues.")

    # Question 2: Direction of greatest variance vs v1
    print("Question 2: The greatest-variance direction follows the main diagonal trend in the scatter; "
          "this corresponds to v_1.")

    # 6) Eigenvalue ratios
    lam_sum = lam1 + lam2
    lam1_ratio = lam1 / lam_sum
    lam2_ratio = lam2 / lam_sum
    print(f"\nlambda_1 ratio: {lam1_ratio:.6f}")
    print(f"lambda_2 ratio: {lam2_ratio:.6f}")

    # Question 3: Compare ratios to sklearn explained_variance_ratio_
    print("Question 3: The eigenvalue ratios match sklearn's explained_variance_ratio_ (up to numeric rounding).")

    # 7) Project onto v1 (1D coordinates along the first principal direction)
    projection = Xc @ v1
    print("\nData projection into 1-D (onto v_1):")
    print(projection)

    # Question 4: Compare to sklearn x_pca[:,0]
    print("Question 4: The manual projection matches sklearn's x_pca[:, 0] up to a possible sign flip.")

    # 8) Plot the 1D projection as points on a number line
    y = np.zeros_like(projection)
    plt.figure(figsize=(8, 1.8))
    plt.scatter(projection, y, s=18, color="#1f77b4", alpha=0.8)
    # Draw a number line
    xmin, xmax = projection.min(), projection.max()
    pad = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
    plt.hlines(0, xmin - pad, xmax + pad, colors="black", linewidth=1)
    plt.yticks([])
    plt.xlabel("Projection onto v1 (1D)")
    plt.tight_layout()
    plt.savefig("projection_1D.png", dpi=150)
    plt.close()

    # Question 5: Compare the 1D plot to the original data
    print("Question 5: The 1D plot preserves ordering and spread along the main variance direction, "
          "but collapses the perpendicular variation.")

if __name__ == "__main__":
    main()