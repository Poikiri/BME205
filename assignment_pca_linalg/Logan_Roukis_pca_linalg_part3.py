import numpy as np
from sklearn.decomposition import NMF
from typing import Dict, List, Tuple


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the Dogs SNP dataset.
    Returns:
        X: (m, d) non-negative matrix of SNP features
        clades: (m,) array of clade labels (strings)
    """
    X = np.load('dogs_X.npy', allow_pickle=True)
    clades = np.load('dogs_clades.npy', allow_pickle=True)
    if X.min() < 0:
        raise ValueError("NMF requires non-negative data, but dogs_X.npy contains negatives.")
    return X, clades


def fit_nmf(X: np.ndarray, n_components: int = 5, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Factor X ≈ W H using NMF.
    Returns:
        W: (m, n_components) non-negative matrix (ancestry proportions before normalization)
        H: (n_components, d) non-negative matrix (ancestry patterns)
    """
    nmf = NMF(
        n_components=n_components,
        init='nndsvda',
        random_state=random_state,
        max_iter=500,
        solver='cd',
    )
    W = nmf.fit_transform(X)
    H = nmf.components_
    return W, H


def row_normalize(W: np.ndarray) -> np.ndarray:
    """
    Row-normalize W so each row sums to 1. Handles zero rows gracefully.
    """
    sums = W.sum(axis=1, keepdims=True)
    # Avoid division by zero; if a row sums to zero, keep it as zeros
    sums_safe = sums.copy()
    sums_safe[sums_safe == 0] = 1.0
    W_norm = W / sums_safe
    return W_norm


def clade_means(W_norm: np.ndarray, clades: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """
    Compute mean normalized ancestry per clade.
    Returns:
        sorted_clades: list of unique clade names, alphabetically sorted
        means: (num_clades, k) array of clade-average ancestry fractions
    """
    unique_clades = sorted(np.unique(clades).tolist())
    means = []
    for clade in unique_clades:
        idx = (clades == clade)
        means.append(W_norm[idx].mean(axis=0))
    return unique_clades, np.vstack(means)


def order_components_by_prevalence_across_clades(clade_means_matrix: np.ndarray) -> np.ndarray:
    """
    Determine a component order by average prevalence across clades (equal weight per clade).
    Returns:
        order: indices of components sorted from most to least prevalent.
    """
    prevalence = clade_means_matrix.mean(axis=0)  # average across clades
    order = np.argsort(prevalence)[::-1]  # descending
    return order


def write_summary_tsv(filepath: str, clades_sorted: List[str], clade_means_matrix: np.ndarray, comp_order: np.ndarray) -> None:
    """
    Write the dogs_ancestry_summary.tsv with required ordering and rounding.
    Columns: clade, ancestry1..ancestry5 ordered by prevalence.
    """
    k = len(comp_order)
    header = ['clade'] + [f'ancestry{i+1}' for i in range(k)]
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\t'.join(header) + '\n')
        for clade, row in zip(clades_sorted, clade_means_matrix):
            ordered = row[comp_order]
            vals = [f"{v:.2f}" for v in ordered]
            f.write('\t'.join([clade] + vals) + '\n')


def main():
    # 1) Load data
    X, clades = load_data()

    # 2) NMF factorization
    W, H = fit_nmf(X, n_components=5, random_state=42)

    # 3) Normalize W rows to proportions
    W_norm = row_normalize(W)

    # 4) Average by clade (alphabetically sorted)
    clades_sorted, means_by_clade = clade_means(W_norm, clades)

    # 5) Determine component order by prevalence across clades (equal-weight per clade)
    comp_order = order_components_by_prevalence_across_clades(means_by_clade)

    # 6) Write summary TSV
    write_summary_tsv(
        filepath='dogs_ancestry_summary.tsv',
        clades_sorted=clades_sorted,
        clade_means_matrix=means_by_clade,
        comp_order=comp_order,
    )

    # Optional: brief stdout confirmation
    print('Saved dogs_ancestry_summary.tsv with', len(clades_sorted), 'clades and', len(comp_order), 'ancestries.')


if __name__ == '__main__':
    main()
