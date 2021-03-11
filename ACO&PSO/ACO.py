# Vitor Diungaro - ACO

import Ant
import plots
import random
import math
import os
import time
import sys
import numpy as np
from scipy.spatial import distance


def read_tsp():
    file = open('berlin52.tsp','r+')

    name = file.readline().split()[1]  # NAME
    file_type = file.readline().split()[1]  # TYPE
    comment = file.readline().strip().split()[1:]  # COMMENT
    dimension = file.readline().split()[1]  # DIMENSION
    edge_weight_type = file.readline().split()[1]  # EDGE_WEIGHT_TYPE
    file.readline()

    cities = []

    for line in file:
        l = line.strip().split()
        if l[0] == 'EOF':
            break
        cities.append([float(l[1]),float(l[2])])

    return {"COMMENT": ' '.join(comment), "DIMENSION": dimension, "TYPE": file_type,
            "EDGE_WEIGHT_TYPE": edge_weight_type, "CITIES": cities}

# Cria a matriz de distância entre os nós (i,j)
# ** somando 1e-12 para evitar divisão por 0
def create_distance_matrix(cities_list):
    n = len(cities_list)
    distance_matrix = np.zeros(shape=(n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = distance.euclidean(
                cities_list[i], cities_list[j]) + 1e-12

    return distance_matrix

# Retorna o custo (distância) de 'route'
def compute_route_distance(route, cities_list):
    d = 0.0

    for i in range(len(route)-1):
        j = i+1

        d += distance.euclidean(cities_list[route[i]],
                                cities_list[route[j]])

    return d

# Função responsável por obter as
# coordenadas (x,y) de cada cidade
def get_cities_xys(cities):
    cities_x = []
    cities_y = []
    for city in cities:
        cities_x.append(city[0])
        cities_y.append(city[1])

    return cities_x, cities_y

# Função para obter as coordenadas da melhor rota obtida
def get_best_route_xys(cities_x, cities_y, best_route):
    best_route_x = []
    best_route_y = []
    for city in best_route:
        best_route_x.append(cities_x[city])
        best_route_y.append(cities_y[city])

    return best_route_x, best_route_y

# *********************** #
#   ACO aplicado ao TSP   #
# *********************** #
def aco_tsp(cities, distance_matrix, max_it, alfa, beta, ro, N, e, Q, tau_0, b):
    # Inicializando tau_ij (feromonio)
    tau_matrix = np.full((e, e), tau_0, dtype=float)

    # Inicializando população de formigas em cidades aleatórias
    ant_population = [Ant.Ant(
        num_nodes=e, position=random.randrange(0, e)) for i in range(N)]

    # Inicializando melhor rota
    best_route = [i for i in range(e)]
    best_route.append(0)
    L_best = compute_route_distance(best_route, cities)

    # Calculando visibilidade
    vis_matrix = 1 / distance_matrix

    for t in range(max_it):
        T_k = [[] for i in range(e)]
        # Construindo uma rota para cada formiga
        for i in range(N):
            a = 0
            T_k[i] = ant_population[i].generate_route(
                tau_matrix, alfa, beta, vis_matrix)

        # Avaliando as rotas construidas
        distances = np.zeros((e))
        for i in range(e):
            distances[i] = compute_route_distance(T_k[i], cities)

        # Encontra a rota com a menor distância e atualiza a melhor rota
        min_route = np.amin(distances)
        min_route_index = np.argmin(distances)
        if L_best > min_route:
            best_route = T_k[min_route_index]
            L_best = round(min_route, 4)

        # Para cada aresta atualiza as trilhas de feromonio
        delta_tau_k = np.zeros((e, e), dtype=float)
        delta_tau_b = np.zeros((e, e), dtype=float)

        # Povoando delta_tau_k
        for k in range(e):
            for i in range(T_k[k].shape[0]-1):
                index_i = T_k[k][i]
                index_j = T_k[k][i+1]
                delta_tau_k[index_i, index_j] = Q/distances[k]

        # Povoando delta_tau_b
        for i in range(best_route.shape[0]-1):
            index_i = best_route[i]
            index_j = best_route[i+1]
            delta_tau_b[index_i, index_j] = b*Q/L_best

        tau_matrix = ((1-ro)*tau_matrix) + delta_tau_k + (b*delta_tau_b)

    return best_route, L_best

# ***************** #
# 		 MAIN       #
# ***************** #
def main():
    try:
        os.makedirs("Resultados Experimentais/")
    except FileExistsError:
        # directory already exists
        pass

    start = time.time()

    tsp = read_tsp()
    cities = tsp["CITIES"]

    exec_times = []
    costs = []

    L_best = sys.maxsize
    for i in range(100):
        # Chamada do ACO
        start_ = time.time()
        best_route_aux, L_best_aux = aco_tsp(cities=cities, distance_matrix=create_distance_matrix(cities), max_it=30, alfa=1, beta=5, ro=0.5, N=len(
            cities), e=len(cities), Q=100, tau_0=1e-06, b=5)
        end_ = time.time()

        # Encontra o melhor caminho dentre todas as iterações
        if L_best > L_best_aux:
            L_best = L_best_aux
            best_route = best_route_aux

        # Adiciona as listas de tempo de execução ao longo do tempo
        # e custo ao longo do tempo os valores da iteração atual do ACO
        exec_times.append(round(end_-start_, 4))
        costs.append(L_best_aux)

    # Preparando dados para gerar os gráficos das cidades
    # e do melhor carminho encontrado
    cities_x, cities_y = get_cities_xys(cities)
    best_route_x, best_route_y = get_best_route_xys(
        cities_x, cities_y, best_route)

    # Gera os gráficos
    plots.plot_cities(cities_x, cities_y,
                file_path="ResultsAco/cidadesBerlin52.pdf")
    plots.plot_best_route(cities_x, cities_y, best_route_x, best_route_y,
                    custo=L_best, file_path="ResultsAco/melhorcaminho.pdf")
    plots.plot_cost_over_time(
        costs, "ResultsAco/custosPorExecução.pdf")
    plots.plot_exec_time_over_time(
        exec_times, "ResultsAco/tempoPorExecução.pdf")

    print("L_best =", L_best)
    print("best_route =", [best_route[i]+1 for i in range(len(best_route))])

    end = time.time()
    print("execution time:", round((end-start), 4), "s.")


if __name__ == "__main__":
    main()


