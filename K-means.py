import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys

def load_data(file_path):
    """
    Load data from a file, excluding the last column the class label
    
    Args:
        file_path: Path to the data file.
    
    Returns:
        data: A numpy array of the data.
    """
    data = np.loadtxt(file_path)
    return data[:, :-1]  

# def euclidean_distance(point1, point2):
#     return np.sqrt(np.sum((point1-point2) **2))

def generate_more_centroids(data, n, random_state):
    """
    Generates a larger number of initial centroids than required to help with the initial centroid problem.
    
    Args:
        data: The data to generate centroids from.
        n: The number of centroids to generate (K * 5)
        random_state: Random seed for reproducibility set to 0
    
    Returns:
        A numpy array of the randomly selected centroids.
    """
    np.random.seed(random_state)
    indices = np.random.choice(data.shape[0], n, replace=False)
    return data[indices]

def select_widely_separated_centroids(k, extra_centroids):
    """
    Selects the k initial centroids that are most widely separated.
    
    Args:
        k: The number of centroids to select.
        extra_centroids: The larger set of centroids generated to select from.
    
    Returns:
        A numpy array of the selected centroids.
    """
    centroids = [extra_centroids[0]]  # Start with the first centroid
    for _ in range(1, k):
        #distances = [min([euclidean_distance(centroid, c) for c in centroids]) for centroid in extra_centroids]
        distances = np.min(np.linalg.norm(extra_centroids[:, np.newaxis] - np.array(centroids), axis=2), axis=1)        
        next_centroid_idx = np.argmax(distances)
        centroids.append(extra_centroids[next_centroid_idx])
    return np.array(centroids)

def assign_clusters(data, centroids):
    """
    Assigns each data point to the nearest centroid.
    
    Args:
        data: The data to assign to clusters.
        centroids: The centroids to assign the data to.
    
    Returns:
        A numpy array of cluster assignments for each data point.
        Using minimum Euclidean distance to assign each data point to a cluster.
    """
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)
    # labels = np.zeros(data.shape[0], dtype=int)
    # for i, point in enumerate(data):
    #     distances = [euclidean_distance(point, centroid) for centroid in centroids]
    #     labels[i] = np.argmin(distances)
    # return labels

def update_centroids(data, labels, k):
    """
    Updates the centroids based on the mean of the data points assigned to each cluster.
    
    Args:
        data: The data points.
        labels: The cluster assignments for each data point.
        k: The number of clusters.
    
    Returns: 
        A numpy array of the updated centroids.
    """
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])

def calculate_error(data, labels, centroids):
    """
    Calculates the error for the clustering following the assignments formula.

    Args:
        data: The data points.
        labels: The cluster assignments for each data point.
        centroids: The centroids of the clusters.
    Returns:
        The sum of squared errors.
    """ 
    error = 0
    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        error += np.sum(np.linalg.norm(cluster_points - centroids[i], axis=1))
    return error

    # sse = sum(np.sum((data[labels == i] - centroids[i]) ** 2) for i in range(len(centroids)))
    # return sse
    # sse = 0
    # for i, centroid in enumerate(centroids):
    #     for point in data[labels==i]:
    #         sse += euclidean_distance(point, centroid) #** 2
    # return sse
def kmeans(data, k, max_iter=20, random_state=0, n_initial_centroids=None):
    """
    Perform k-means clustering on the data with possible solution to the initial centroid problem.
    
    Args:
        data: The data to cluster.
        k: The number of clusters.
        max_iter: The maximum number of iterations to run.
        random_state: Random seed for reproducibility set to 0.
        n_initial_centroids: The number of initial centroids to generate.
    
    Returns:
        labels: The cluster assignments for each data point.
        centroids: The centroids of the clusters.
        sse: The sum of squared errors.
    """
    if n_initial_centroids is None:
        n_initial_centroids = k * 5  # Default to 3 times the number of desired clusters

    extra_centroids = generate_more_centroids(data, n_initial_centroids, random_state)
    centroids = select_widely_separated_centroids(k, extra_centroids)

    for _ in range(max_iter):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels, centroids, calculate_error(data, labels, centroids)

def plot_clusters_with_pca(data, labels, centroids, k):
    """
    Plot the clusters of the data using PCA for dimensionality reduction.
    
    Args:
        data: The data points.
        labels: The cluster assignments for each data point.
        centroids: The centroids of the clusters.
        k: The number of clusters.
    
    Returns:
        None, displays plot."""
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    reduced_centroids = pca.transform(centroids)

    plt.figure()
    for i in range(k):
        cluster_data = reduced_data[labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i + 1}')
    plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], color='black', marker='x', s=100, label='Centroids')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title(f'K-Means Clustering with PCA Projection, K={k}')
    plt.legend()
    plt.show()

def main(file_path):
    """
    Load data and run kmeans for multiple values of k and plot the results (optional).
    
    Args: 
        file_path: Path to the data file.
    
    Returns:
        None, displays plots.
    """
    data = load_data(file_path)
    k_values = range(2, 11)
    errors = []

    for k in k_values:
        labels, centroids, err = kmeans(data, k)
        errors.append(err)
        print(f"K={k}, Error={err:.4f}")
        plot_clusters_with_pca(data, labels, centroids, k)

    plt.figure()
    plt.plot(k_values, errors, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Error')
    plt.title('K-Means Clustering Error')
    plt.show()

if __name__ == "__main__":
    file_path = sys.argv[1]
    main(file_path)