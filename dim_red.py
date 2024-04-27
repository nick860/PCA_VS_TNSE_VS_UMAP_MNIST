import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap

def plot_pca_2d(reduced_data, labels,technique):
    """
    Visualizes the data in 2D using PCA and colors points by digit label.

    Args:
        reduced_data (numpy.ndarray): The data with reduced dimensions.
        labels (torch.Tensor): The labels for each data point.
    """

    plt.figure(figsize=(8, 6)) # this is to set the size of the plot
    colors = ['red', 'yellow', 'green', 'blue', 'purple', 'orange', 'brown', 'gray', 'pink', 'cyan']
    the_digits = range(10)
    #the_digits = [1,6]
    for i in the_digits:
        # Filter data points for each digit
        data_i = reduced_data[labels == i] 
        plt.scatter(data_i[:, 0], data_i[:, 1], label=i, c=colors[i]) # x=PC1, y=PC2

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'{technique} Visualization of MNIST Dataset')
    plt.legend()
    plt.show()

# Define data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization values
])

def PCA_Visualization(all_images_np, all_labels):
    # Apply PCA with 95% variance retained (adjust as needed)
    pca = PCA(n_components=2)  # You can also set a specific number of components
    pca.fit(all_images_np) # Fit the PCA model with the data
    # Get the transformed data with reduced dimensions
    reduced_data = pca.transform(all_images_np) # shape will be (60000, 2)
    plot_pca_2d(reduced_data, torch.from_numpy(all_labels), technique='PCA')


def TSNE_Visualization(all_images_np, all_labels):
    # Apply t-SNE with 2 components
    tsne = TSNE(n_components=2, perplexity=30) # perplexity is the number of nearest neighbors that is used in other manifold learning algorithms
    reduced_data = tsne.fit_transform(all_images_np)
    plot_pca_2d(reduced_data, torch.from_numpy(all_labels), technique='t-SNE')

def UMAP_Visualization(all_images_np, all_labels):
    # Apply UMAP with 2 components
    reducer = umap.UMAP() # the default number of components is 2
    reduced_data = reducer.fit_transform(all_images_np)
    plot_pca_2d(reduced_data, torch.from_numpy(all_labels), technique='UMAP')

if __name__ == "__main__":
    # Load MNIST dataset
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    # Extract flattened images from the dataset
    # Extract flattened images and labels
    all_images = []
    all_labels = []
    for images, labels in train_loader:
        flattened_images = images.view(images.size(0), -1)
        all_images.append(flattened_images.cpu().numpy()) 
        all_labels.append(labels.cpu().numpy())

    # all_images will be in shape of (num_batches, batch_size, 784)
    # all_labels will be in shape of (num_batches, batch_size)

    # Concatenate all flattened images into a single numpy array
    all_images_np = np.concatenate(all_images, axis=0) # after concarination shape will be (num_batches*batch_size, 784) that is (60000, 784) 
    all_labels = np.concatenate(all_labels, axis=0) # after concarination shape will be (num_batches*batch_size) that is (60000)

    print("Processing PCA Visualization")
    #PCA_Visualization(all_images_np, all_labels)
    print("Processing TSNE Visualization")
    #TSNE_Visualization(all_images_np, all_labels)
    print("Processing UMAP Visualization")
    UMAP_Visualization(all_images_np, all_labels)
