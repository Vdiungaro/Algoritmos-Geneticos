import statistics
import os
import matplotlib.pyplot as plt
import numpy as np

def stats(bapt_mean, best_apt_value, apt_mean, mean_apt_value, gen_value_total,gen_value,exec_time,gen_time):
    gen_value_total.append(gen_value)
    bapt_mean.append(statistics.mean(best_apt_value))
    apt_mean.append(statistics.mean(mean_apt_value))
    gen_time.append(exec_time)
    return bapt_mean,apt_mean,gen_value_total, gen_time

def write(aux1,aux2,file_path):
    z1 = np.nanmean(aux1)
    z2 = np.nanmean(aux2)
    file = open(str(file_path), 'w+')
    file.write(z1)
    file.write(z2)
    #file.write('Tempo medio de execução por geração: %4.2f' % round(statistics.mean(gen_time), 2) + '\n')
    file.close()


def plot_gen(gen_num,file_path):
    figure_2 = plt.figure()
    plt.plot(gen_num, 'r', label='Numero de Gerações')
    plt.title('Numero de Gerações para atingir o melhor individuo')
    plt.xlabel('Execução')
    plt.ylabel('Numero de Gerações')
    plt.legend()
    figure_2.savefig(str(file_path))
    plt.show()

def plot_bestmean(best_so_far,file_path,title):
    figure_3 = plt.figure()
    plt.plot(best_so_far, 'gold')
    plt.title(title)
    plt.xlabel('Iteração')
    plt.ylabel('média do melhor')
    figure_3.savefig(str(file_path))
    plt.show()

def plot_aptmean(apt_mean,file_path,title):
    figure_4 = plt.figure()
    plt.plot(apt_mean, 'gold')
    plt.title(title)
    plt.xlabel('Iteração')
    plt.ylabel('média f(x,y)')
    figure_4.savefig(str(file_path))
    plt.show()

def plot_time(gen_time,file_path,):
    figure_1 = plt.figure()
    plt.plot(gen_time, 'y', label='Tempo de execução')
    plt.title("Tempo de execução em cada execução")
    plt.xlabel('Execução')
    plt.ylabel('Tempo de execução (s)')
    plt.legend()
    figure_1.savefig(str(file_path))
    plt.show()