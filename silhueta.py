import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    data = df.iloc[:, :-1].values  # A última coluna foi removida pois era a classe
    normalized_data = (data - data.mean(axis=0)) / data.std(axis=0)  # Normalização dos dados
    return normalized_data

def input_int(message): #Garante que o K informado pelo usuário sempre será um número inteiro
    while True:
        try:
            value = int(input(message))
            return value
        except ValueError:
            print("Por favor, insira um número inteiro válido.")


def initialize_centroids(data, K):
    np.random.seed(42)
    centroids = data[np.random.choice(data.shape[0], K, replace=False)]  # Inicializando o centroide de forma aleatória
    return centroids

class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.centroids = initialize_centroids(X, self.n_clusters)
        
        for i in range(self.max_iter):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))  # Distância Euclidiana
            labels = np.argmin(distances, axis=0)
            
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
                
        self.labels_ = labels
        return labels

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point2 - point1) ** 2))

def calculate_distance_matrix(data):
    n = len(data)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            distances[i, j] = euclidean_distance(data[i], data[j])
            distances[j, i] = distances[i, j]
    
    return distances

def single_link(data, n_clusters):
    n = len(data)
    clusters = [[i] for i in range(n)]
    distances = calculate_distance_matrix(data)
    cluster_map = {i: i for i in range(n)}
    cluster_label = n
    
    while len(clusters) > n_clusters:
        min_distance = np.inf
        merge_indices = None
        
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                for idx1 in clusters[i]:
                    for idx2 in clusters[j]:
                        if distances[idx1, idx2] < min_distance:
                            min_distance = distances[idx1, idx2]
                            merge_indices = (i, j)
        
        cluster1, cluster2 = clusters[merge_indices[0]], clusters[merge_indices[1]]
        new_cluster = cluster1 + cluster2
        clusters.append(new_cluster)

        cluster1_id = cluster_map[cluster1[0]]
        cluster2_id = cluster_map[cluster2[0]]
        
        for idx in new_cluster:
            cluster_map[idx] = cluster_label
        cluster_label += 1
        
        del clusters[max(merge_indices)]
        del clusters[min(merge_indices)]
    
    final_labels = np.zeros(n, dtype=int)
    for cluster_index, cluster in enumerate(clusters):
        for idx in cluster:
            final_labels[idx] = cluster_index
    
    return final_labels

#a.	Implemente a seguinte medida:
#   i.	Silhueta Simplificada

def calculate_silhouette(X, labels):
    n_clusters = len(np.unique(labels))
    silhouette_vals = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        same_cluster = X[labels == labels[i]]
        other_clusters = [X[labels == label] for label in np.unique(labels) if label != labels[i]]

        if len(same_cluster) > 1:
            a_i = np.mean([euclidean_distance(X[i], point) for point in same_cluster if not np.array_equal(X[i], point)])
        else:
            a_i = 0

        if other_clusters:
            b_i = np.min([np.mean([euclidean_distance(X[i], point) for point in cluster]) for cluster in other_clusters if len(cluster) > 0])
        else:
            b_i = 0

        silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0 #B-A/max(A,B)
    
    return np.mean(silhouette_vals)

def write_result(filename, kmeans_labels, single_link_labels, kmeans_silhouette, single_link_silhouette, best_algorithm):

    with open(filename, 'w') as f:
        f.write(f"K-means Silhouette Score: {kmeans_silhouette}\n")
        f.write(f"Single Link Silhouette Score: {single_link_silhouette}\n")
        f.write(f"Melhor algoritmo: {best_algorithm}\n\n")

        f.write("K-means Clusters:\n")
        kmeans_clusters = {i: np.where(kmeans_labels == i)[0].tolist() for i in np.unique(kmeans_labels)}
        for cluster in kmeans_clusters.values():
            f.write(f"{set(cluster)}\n")
        
        f.write("\nSingle Link Clusters:\n")
        single_link_clusters = {i: np.where(single_link_labels == i)[0].tolist() for i in np.unique(single_link_labels)}
        for cluster in single_link_clusters.values():
            f.write(f"{set(cluster)}\n")

def main():
    file_path = "iris.data"
    data = preprocess_data(file_path) #e.	Lembre-se de tirar a classe da base de dados (caso ela esteja disponível) antes de executar os algoritmos de agrupamento.


    # d.	O valor de k será informado pelo usuário, bem como a base de dados a ser usada.
    k = input_int("Digite a quantidade de clusters para o agrupamento:")

    # b.	Execute o algoritmo K-Means (já implementado), usando uma base de dados pública. 
    kmeans = KMeans(n_clusters=k)
    kmeans_labels = kmeans.fit(data)
    kmeans_silhouette = calculate_silhouette(data, kmeans_labels)

    # c.	Execute o algoritmo Single Link (já implementado), usando a mesma base de dados pública da questão a.
    single_link_labels = single_link(data, k) #i.	Para o Single Link, pegue a partição gerada com a mesma quantidade de grupos do K-Means. 
    single_link_silhouette = calculate_silhouette(data, single_link_labels)


    #g.	Imprima o resultado da medida de validação para a execução do algoritmo K-Means e para o algoritmo Single Link. 
    print(f'K-means Silhouette Score: {kmeans_silhouette}')
    print(f'Single Link Silhouette Score: {single_link_silhouette}')

    # Determinando o melhor algoritmo
    if kmeans_silhouette > single_link_silhouette:
        best_algorithm = "K-means"
    else:
        best_algorithm = "Single Link"

    #h.	Escreva na tela qual algoritmo apresentou melhor resultado. 
    print(f"Algoritmo com melhor resultado: {best_algorithm}")
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file = f"resultado_{current_time}.txt"
    write_result(result_file, kmeans_labels, single_link_labels, kmeans_silhouette, single_link_silhouette, best_algorithm)
    

if __name__ == "__main__":
    main()
