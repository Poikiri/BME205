# Copilot Instructions for PCA/Linear Algebra Assignment

## Project Overview
This is a **structured educational assignment** focused on dimensionality reduction techniques (PCA and NMF) with three distinct parts that must be implemented as separate Python scripts. Each part builds mathematical intuition through hands-on implementation and visualization.

## Critical Architecture & Data Flow

### Three-Part Structure (Independent Scripts)
- **Part 1**: Manual PCA implementation on 2D synthetic data (`part1_data.npy`) → tutorial with stdout questions + `projection_1D.png`
- **Part 2**: sklearn PCA on MNIST subset → 2D visualization + image reconstruction (`MNIST_*.png` outputs)  
- **Part 3**: NMF on dog SNP dataset → ancestry analysis (`dogs_ancestry_summary.tsv`)

### Key Data Dependencies
```
part1_data.npy          # 2D synthetic dataset for PCA tutorial
MNIST_X_subset.npy      # 6000 images (600 per digit 0-9), 784 features
MNIST_y_subset.npy      # Corresponding labels  
dogs_X.npy              # 1355 dogs × 784 SNPs
dogs_clades.npy         # 30 dog breed clades
```

## Essential Developer Workflows

### Environment Setup (Required First)
```bash
conda env create -f pca_linalg.yaml
conda activate pca_linalg
```
**Fixed dependencies**: numpy==2.1.1, scikit-learn==1.5.2, matplotlib==3.9.2

### Execution Pattern (Naming Convention Critical)
```bash
python Firstname_Lastname_pca_linalg_part1.py  # Must print to stdout + save projection_1D.png
python Firstname_Lastname_pca_linalg_part2.py  # Saves 4 images: MNIST_PCA_2D.png, etc.
python Firstname_Lastname_pca_linalg_part3.py  # Saves dogs_ancestry_summary.tsv
```

## Project-Specific Patterns

### Part 1: Manual PCA Implementation Pattern
- **Dual approach required**: Compare manual eigendecomposition with sklearn PCA
- **Stdout format**: Must print specific values (eigenvals, eigenvecs, ratios) + answer 5 questions
- **Math sequence**: Center data → Covariance matrix → Eigendecomposition → Project onto v_1
- See `part1_example_stdout.txt` for required output structure

### Part 2: MNIST Visualization & Reconstruction
- **Standard flow**: Load → PCA(n_components=2) → Scatter plot → Image reconstruction
- **Critical**: Use `inverse_transform()` for reconstruction from reduced space
- **Generative aspect**: Manual coordinate selection in 2D PCA space to synthesize digit-like images

### Part 3: NMF Ancestry Analysis  
- **Matrix factorization**: X ≈ WH where W = ancestry proportions, H = ancestry patterns
- **Normalization critical**: Row-normalize W so individual ancestry components sum to 1
- **Output format strict**: TSV with clades alphabetically sorted, ancestry columns by prevalence

### File Output Conventions
- **Images**: Save with exact filenames (`projection_1D.png`, `MNIST_PCA_2D.png`, etc.)
- **TSV format**: Tab-separated, 2 decimal places, specific column ordering by prevalence
- **All outputs**: Must be generated fresh by script execution (no pre-existing files)

## Common Implementation Gotchas
- **Data loading**: Always use `np.load('file.npy', allow_pickle=True)` for compatibility
- **Eigenvector ordering**: `np.linalg.eigh()` returns ascending eigenvals - need to handle PC ordering
- **NMF normalization**: Row normalize W matrix, then average by clade for final summary
- **Image reconstruction**: MNIST images need reshaping (784,) ↔ (28,28) for visualization
- **Coordinate selection**: Part 2 requires manual 2D point selection from PCA visualization

This assignment emphasizes **mathematical understanding through implementation** rather than black-box usage of dimensionality reduction techniques.