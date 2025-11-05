import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the MNIST subset
print("Loading MNIST data...")
X = np.load('MNIST_X_subset.npy', allow_pickle=True)
y = np.load('MNIST_y_subset.npy', allow_pickle=True)

print(f"Dataset shape: {X.shape}")  # Should be (6000, 784)
print(f"Labels shape: {y.shape}")   # Should be (6000,)

# Step 1: Perform PCA to reduce from 784 dimensions to 2 dimensions
print("\nPerforming PCA to reduce to 2 dimensions...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Transformed data shape: {X_pca.shape}")  # Should be (6000, 2)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# Step 2: Visualize the 2D PCA-transformed data
print("\nCreating 2D visualization...")
plt.figure(figsize=(12, 10))

# Create a scatter plot with different colors for each digit
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for digit in range(10):
    mask = y == digit
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=[colors[digit]], label=f'Digit {digit}', 
                alpha=0.6, s=20)

plt.xlabel('First Principal Component', fontsize=12)
plt.ylabel('Second Principal Component', fontsize=12)
plt.title('MNIST Digits in 2D PCA Space', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('MNIST_PCA_2D.png', dpi=150)
print("Saved: MNIST_PCA_2D.png")
plt.close()

# Step 3: Reconstruct an image using the first 2 principal components
print("\nReconstructing first image from 2 PCs...")

# Get the first image (index 0)
original_image = X[0]
print(f"Original image label: {y[0]}")

# Transform to 2D PCA space
reduced_representation = pca.transform(original_image.reshape(1, -1))
print(f"Reduced representation (2D): {reduced_representation}")

# Reconstruct back to 784 dimensions
reconstructed_image = pca.inverse_transform(reduced_representation)

# Visualize original image
plt.figure(figsize=(6, 6))
plt.imshow(original_image.reshape(28, 28), cmap='gray')
plt.title(f'Original Image (Digit {y[0]})', fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('MNIST_original.png', dpi=150)
print("Saved: MNIST_original.png")
plt.close()

# Visualize reconstructed image
plt.figure(figsize=(6, 6))
plt.imshow(reconstructed_image.reshape(28, 28), cmap='gray')
plt.title(f'Reconstructed from 2 PCs (Digit {y[0]})', fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('MNIST_reconstructed_2PC.png', dpi=150)
print("Saved: MNIST_reconstructed_2PC.png")
plt.close()

# Step 4: Generate a new "1" digit by selecting coordinates in PCA space
print("\nGenerating a new '1' digit from selected coordinates...")

# Look at the PCA plot and select coordinates where digit '1' appears
# From the visualization, digit 1s typically cluster in a specific region
# Let's select a coordinate in that region
# (You may need to adjust these values based on your actual plot)
selected_coords = np.array([[-900, 500]])  # Adjust based on your PCA plot

print(f"Selected 2D coordinates: {selected_coords}")

# Reconstruct this point back to 784 dimensions
generated_image = pca.inverse_transform(selected_coords)

# Visualize the generated image
plt.figure(figsize=(6, 6))
plt.imshow(generated_image.reshape(28, 28), cmap='gray')
plt.title('Generated Digit "1" from Selected Coordinates', fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('MNIST_reconstructed_1_from_coord.png', dpi=150)
print("Saved: MNIST_reconstructed_1_from_coord.png")
plt.close()

print("\n✓ Part 2 complete! All images saved successfully.")
