import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix


def knp_clustering(X, K):
    # Вычисляем матрицу расстояний между образцами
    dist_matrix = distance_matrix(X, X)
    np.fill_diagonal(dist_matrix, np.inf)  # Исключаем нулевые расстояния до самих себя

    G = nx.Graph()
    for i in range(len(X)):
        G.add_node(i)

    # Найти пару точек с минимальным расстоянием и соединить их
    i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    G.add_edge(i, j, weight=dist_matrix[i, j])

    # Пока остаются изолированные точки
    while len(list(nx.isolates(G))) > 0:
        # Находим ближайшую изолированную точку к уже соединённым точкам
        isolates = list(nx.isolates(G))
        non_isolates = list(set(G.nodes) - set(isolates))

        min_dist = np.inf
        closest_pair = (None, None)
        for iso in isolates:
            for node in non_isolates:
                if dist_matrix[iso, node] < min_dist:
                    min_dist = dist_matrix[iso, node]
                    closest_pair = (iso, node)

        # Соединяем изолированную точку с ближайшей к ней не изолированной
        G.add_edge(closest_pair[0], closest_pair[1], weight=min_dist)

    # Удалить K-1 самых длинных рёбер
    edges_sorted_by_weight = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    for i in range(K - 1):
        edge_to_remove = edges_sorted_by_weight[i]
        G.remove_edge(edge_to_remove[0], edge_to_remove[1])

    # Получаем кластеры как связные компоненты графа
    clusters = list(nx.connected_components(G))
    return clusters


from sklearn.datasets import make_moons
X, y = make_moons(n_samples=150,
                  noise=0.076)


clusters = knp_clustering(X=X, K=2)

for i, cluster in enumerate(clusters):
    print(f"Кластер {i + 1}: {cluster}")


import matplotlib.pyplot as plt
import time

for i in range(100):
    X, y = make_moons(n_samples=100,
                      noise=0.076,
                      random_state=int(i + time.time()) % 1000001)

    clusters = knp_clustering(X=X, K=2)

    plt.scatter(X[:, 0],
                X[:, 1],
                c='black',
                marker='o',
                edgecolor='black',
                s=50)
    plt.grid()
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


    colors = [
        "#1f77b4",  # синий
        "#ff7f0e",  # оранжевый
        "#2ca02c",  # зеленый
        "#d62728",  # красный
        "#9467bd",  # фиолетовый
        "#8c564b",  # коричневый
        "#e377c2",  # розовый
        "#7f7f7f",  # серый
        "#bcbd22",  # желто-зеленый
        "#17becf",  # голубой
    ]

    for i in range(len(clusters)):
        print(list(clusters[i]))
        plt.scatter(X[list(clusters[i]), 0],
                    X[list(clusters[i]), 1],
                    c=colors[i],
                    marker='o',
                    edgecolor='black',
                    s=50)

    plt.grid()
    plt.title(f"KNP, components number = {len(clusters)}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()
    time.sleep(1)