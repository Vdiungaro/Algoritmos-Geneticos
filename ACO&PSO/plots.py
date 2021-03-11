# Vitor Diungaro - Funções para graficos do ACO

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

plt.style.use('ggplot')

def plot_cities(cities_x, cities_y, file_path):
    f1 = plt.figure()
    plt.plot(cities_x, cities_y, 'o', label="Cidade", color='C1')
    plt.title("Lista de cidades no arquivo berlin52.tsp",
              fontweight='bold', fontsize=12)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    f1.savefig(str(file_path), format='pdf', bbox_inches='tight')


def plot_best_route(cities_x, cities_y, best_route_x, best_route_y, custo, file_path):
    f1 = plt.figure()
    plt.plot(best_route_x, best_route_y, color='C3', path_effects=[
             pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
    plt.plot(cities_x, cities_y, 'o', label="Cidade", color='C1')
    plt.title("Melhor caminho encontrado em todas as execuções\n" +
              "(" + r"$\bf{Custo=}$" + str(custo) + ")", fontweight='bold', fontsize=10)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    f1.savefig(str(file_path), format='pdf', bbox_inches='tight')


def plot_cost_over_time(costs, file_path):
    f1 = plt.figure()
    plt.plot(costs, color='C1', path_effects=[
             pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
    plt.title("Custo mínimo obtido ao longo das execuções",
              fontsize=10, fontweight='bold')
    plt.xlabel("Iteração", fontsize=10)
    plt.ylabel("Custo", fontsize=10)
    f1.savefig(str(file_path), format='pdf', bbox_inches='tight')


def plot_exec_time_over_time(exec_times, file_path):
    f1 = plt.figure()
    plt.plot(exec_times, color='C1', path_effects=[
             pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
    plt.title("Tempo necessário para executar o algoritmo ACO ao longo das execuções",
              fontsize=10, fontweight='bold')
    plt.xlabel("Iteração", fontsize=10)
    plt.ylabel("Tempo de execução (s)", fontsize=10)
    f1.savefig(str(file_path), format='pdf', bbox_inches='tight')
