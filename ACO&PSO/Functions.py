# Vitor Diungaros - Funções para o PSO

import statistics
import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
plt.style.use('ggplot')


# ************************************************************* #
# Funções para plotar dados e escrever arquivo com estatísticas #
# ************************************************************* #


def plot_fs(best_aptitude_values, mean_aptitude_values, population_size, file_path):
    f1 = plt.figure()
    plt.plot(best_aptitude_values, label='f(x,y) mínimo', path_effects=[
        pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
    plt.plot(mean_aptitude_values, label='f(x,y) médio', path_effects=[
        pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
    plt.title("f(x,y) mínimo e médio obtido ao longo das iterações\n" +
              "(" + r"$\bf{qtd\_particulas=}$" + str(population_size) + ")", fontsize=10)
    plt.xlabel("Iteração", fontsize=10)
    plt.ylabel("f(x,y)", fontsize=10)
    plt.legend(fontsize='x-small')
    f1.savefig(str(file_path), format='pdf', bbox_inches='tight')


def plot_fxy_over_time(num_iter_total, population_size, file_path):
    f1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        nrows=5, ncols=1, sharex=True, sharey=True)

    ax1 = plt.subplot(511)
    ax1.plot(num_iter_total[0], label="qtd_particulas=" + str(population_size[0]), color='C0',
             path_effects=[pe.Stroke(linewidth=2, foreground='black'),
                           pe.Normal()])
    ax1.set_title("Quantidade necessária de iterações para minimizar f(x,y) por execução",
                  fontsize=10, fontweight='bold')
    ax1.legend(fontsize='x-small', loc=1)
    ax1.set_yticks(np.arange(0, 400, 100))

    ax2 = plt.subplot(512, sharey=ax1)
    ax2.plot(num_iter_total[1], label="qtd_particulas=" + str(population_size[1]), color='C1',
             path_effects=[pe.Stroke(linewidth=2, foreground='black'),
                           pe.Normal()])
    ax2.legend(fontsize='x-small', loc=1)

    ax3 = plt.subplot(513, sharey=ax1)
    pl.plot(num_iter_total[2], label="qtd_particulas=" + str(population_size[2]), color='C2',
            path_effects=[pe.Stroke(linewidth=2, foreground='black'),
                          pe.Normal()])
    ax3.legend(fontsize='x-small', loc=1)

    ax4 = plt.subplot(514, sharey=ax1)
    ax4.plot(num_iter_total[3], label="qtd_particulas=" + str(population_size[3]), color='C3',
             path_effects=[pe.Stroke(linewidth=2, foreground='black'),
                           pe.Normal()])
    ax4.legend(fontsize='x-small', loc=1)

    ax5 = plt.subplot(515, sharey=ax1)
    ax5.plot(num_iter_total[4], label="qtd_particulas=" + str(population_size[4]), color='C4',
             path_effects=[pe.Stroke(linewidth=2, foreground='black'),
                           pe.Normal()])
    ax5.legend(fontsize='x-small', loc=1)
    ax5.set_xlabel('Número de Execuções', fontsize=10)

    # for i in range(0, len(num_iter_total)):
    #    plt.plot(num_iter_total[i], label="qtd_particulas=" + str(population_size[i]),
    # path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])

    f1.tight_layout()
    f1.text(-0.01, 0.5, 'Iterações (s)', va='center',
            rotation='vertical', fontsize=10)
    f1.savefig(str(file_path), format='pdf', bbox_inches='tight')


def plot_execution_time_over_time(exec_time, population_size, file_path):
    f2, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        nrows=5, ncols=1, sharex=True, sharey=True)

    ax1 = plt.subplot(511)
    ax1.plot(exec_time[0], label="qtd_particulas=" + str(population_size[0]), color='C0',
             path_effects=[pe.Stroke(linewidth=2, foreground='black'),
                           pe.Normal()])
    ax1.set_title("Tempo necessário para minimizar f(x,y) por execução",
                  fontsize=10, fontweight='bold')
    ax1.legend(fontsize='x-small', loc=1)
    ax1.set_yticks(np.arange(0, 0.3, 0.1))

    ax2 = plt.subplot(512, sharey=ax1)
    ax2.plot(exec_time[1], label="qtd_particulas=" + str(population_size[1]), color='C1',
             path_effects=[pe.Stroke(linewidth=2, foreground='black'),
                           pe.Normal()])
    ax2.legend(fontsize='x-small', loc=1)

    ax3 = plt.subplot(513, sharey=ax1)
    pl.plot(exec_time[2], label="qtd_particulas=" + str(population_size[2]), color='C2',
            path_effects=[pe.Stroke(linewidth=2, foreground='black'),
                          pe.Normal()])
    ax3.legend(fontsize='x-small', loc=1)

    ax4 = plt.subplot(514, sharey=ax1)
    ax4.plot(exec_time[3], label="qtd_particulas=" + str(population_size[3]), color='C3',
             path_effects=[pe.Stroke(linewidth=2, foreground='black'),
                           pe.Normal()])
    ax4.legend(fontsize='x-small', loc=1)

    ax5 = plt.subplot(515, sharey=ax1)
    ax5.plot(exec_time[4], label="qtd_particulas=" + str(population_size[4]), color='C4',
             path_effects=[pe.Stroke(linewidth=2, foreground='black'),
                           pe.Normal()])
    ax5.legend(fontsize='x-small', loc=1)
    ax5.set_xlabel('Número de Execuções', fontsize=10)

    # f2.text(0.5, 0.04, 'Número de Execuções', ha='center', va='center')
    # f2.text(0.06, 0.5, 'Tempo de Execução (s)',
    #        ha='center', va='center', rotation='vertical')

    # for i in range(2):
    #    for j in range(6):
    #        ax.plot(x, exec_time[j], label="qtd_particulas=" + str(population_size[i]),
    #                path_effects=[pe.Stroke(linewidth=2, foreground='black'),
    #                              pe.Normal()])

    f2.tight_layout()
    f2.text(-0.01, 0.5, 'Tempo de Execução (s)',
            va='center', rotation='vertical', fontsize=10)
    f2.savefig(str(file_path), format='pdf', bbox_inches='tight')


def write_statistics(pop_size, fs_med_mean, num_iters, exec_times, file_path):
    # primeira iteração (10 particulas) cria o arquivo de saída
    if pop_size == 10:
        file = open(str(file_path), 'w+')
    else:
        file = open(str(file_path), 'a+')

    file.write(
        "****************************************************************" + "\n")
    file.write("Quantidade de particulas=" + str(pop_size) + "\n")
    file.write("Média das médias de f(x,y) a cada execução: %.2f" %
               round(statistics.mean(fs_med_mean), 2) + "\n")
    file.write("Desvio padrão das médias de médias de f(x,y): %.2f" %
               round(statistics.stdev(fs_med_mean), 2) + "\n")
    file.write("Numero médio de iterações para minimizar f(x,y): %.2f" %
               round(statistics.mean(num_iters), 2) + "\n")
    file.write("Desvio padrão do número medio de iterações: %.2f" %
               round(statistics.stdev(num_iters), 2) + "\n")
    file.write("Tempo médio de execução por iteração: %f" %
               round(statistics.mean(exec_times), 6) + " segundos\n")

    file.close()


# ********************************************************


def append_values(best_aptitude_mean, mean_of_mean_aptitude, best_aptitude_values, mean_aptitude_values, num_gen_total,
                  num_generations, exec_time, start, end):
    best_aptitude_mean.append(statistics.mean(best_aptitude_values))
    mean_of_mean_aptitude.append(statistics.mean(mean_aptitude_values))
    num_gen_total.append(num_generations)
    exec_time.append(end - start)

    return best_aptitude_mean, mean_of_mean_aptitude, num_gen_total, exec_time


def append_num_iters(num_iter_totalA, num_iter_totalB, num_iter_totalC, num_iter_totalD, num_iter_totalE):
    num_iters = []

    num_iters.append(num_iter_totalA)
    num_iters.append(num_iter_totalB)
    num_iters.append(num_iter_totalC)
    num_iters.append(num_iter_totalD)
    num_iters.append(num_iter_totalE)

    return num_iters


def append_exec_times(exec_timeA, exec_timeB, exec_timeC, exec_timeD, exec_timeE):
    exec_times = []

    exec_times.append(exec_timeA)
    exec_times.append(exec_timeB)
    exec_times.append(exec_timeC)
    exec_times.append(exec_timeD)
    exec_times.append(exec_timeE)

    return exec_times
