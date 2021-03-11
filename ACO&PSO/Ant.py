# Vitor Diungaro - Classe de Formigas para o ACO

import numpy as np


class Ant:
    def __init__(self, num_nodes, position):
        self.num_nodes = num_nodes
        self.initial_position = position

    def generate_route(self, tau_matrix, alfa, beta, vis_matrix):
        # Inicializando posição inicial da formiga
        current_position = self.initial_position

        # Criando a lista de nós possíveis para visitar
        node_list = np.ones(self.num_nodes)
        node_list[self.initial_position] = 0

        # Rota T_k
        T_k = []
        T_k.append(current_position)
        # Vetor de probabilidades de transição
        prob_mov = np.zeros(self.num_nodes)

        for _ in range(self.num_nodes - 1):
            # Calculo da probabilidade de transição
            sum = 0.0
            for i in range(self.num_nodes):
                if node_list[i] == 1:
                    sum += (tau_matrix[current_position, i]) ** alfa * \
                           (vis_matrix[current_position, i]) ** beta

            for i in range(self.num_nodes):
                if node_list[i] == 1:
                    prob_mov[i] = ((tau_matrix[current_position, i]) **
                                   alfa * (vis_matrix[current_position, i]) ** beta) / sum
                else:
                    prob_mov[i] = 0

            # Encontra a proxima cidade a ser visitada de acordo com a probabilidade de transição
            next_city = choose_movement(prob_mov)
            current_position = next_city
            node_list[next_city] = 0

            T_k.append(choose_movement(prob_mov))

        # Caso tenha percorrido todos os nós, volta ao inicial
        if 1.0 not in node_list:
            T_k.append(self.initial_position)

        return np.array(T_k)


def choose_movement(prob_mov):
    return np.argmax(prob_mov)
