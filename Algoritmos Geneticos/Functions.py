import numpy as np
import random
import statistics
import os
import math
import matplotlib.pyplot as plt


# --------------------------------------------
# Funções utilizadas por todos os exercicios
# --------------------------------------------
# Função que executa o crossover de pos até (s) do indviduo i com i+1
def crossover(P,pos,s,i):
    aux = P[i+1,pos:s].copy()
    P[i+1,pos:s] = P[i,pos:s]
    P[i, pos:s] = aux
    return P

# Função que faz a mutação em determinada posição do individuo
def mutation(P, j, i):
    P[i,j] = random.randint(0,1)
    return P

def reproduction(P, cc, mc, s):
    pop_size = P.shape[0]

    #Crossover
    for i in range(0,pop_size,3):
        c = round(random.random(),3)
        if c <= cc:
            pos = random.randint(1,s-1)
            if i != pop_size-1:
                P = crossover(P,pos,s,i)
            else:
                P = crossover(P,pos,s,i-1)

    #Mutação
    for i in range(pop_size):
        for j in range(s):
            c = round(random.random(),3)
            if c <= mc:
                P = mutation(P, j, i)

    return P
# Retorna o indice do individuo escolhido no intervalo [0:360]
def roullete(apt_ind):
    aux = 0
    k = 0
    r = random.uniform(0,360)

    for i in range(apt_ind.size):
        k += apt_ind[i]
        if r <= k:
                aux = i
                break
    return aux

def SelectionbyRoulette(pop, apt):
    pop_size = pop.shape[0]
    # Atribui uma porção da roleta para o individuo baseado na sua aptdidão
    apt_ind = np.array([(360*apt[i])/(apt.sum()) for i in range(pop_size)])
    # Gera uma nova população baseado na roleta
    P = np.array([pop[roullete(apt_ind)] for i in range (pop_size)])

    return P

def SelectionByTournament(pop, apt, num_selected):
    pop_size = pop.shape[0]
    # Atribui uma porção da roleta para o individuo baseado na sua aptdidão
    apt_ind = np.array([(360 * apt[i]) / (apt.sum()) for i in range(pop_size)])
    # Seleciona os individuos
    selected = np.zeros(shape=(pop_size,num_selected), dtype=int)

    # Preenche com os possiveis canditados
    for i in range(pop_size):
        for j in range(num_selected):
            selected[i][j] = roullete(apt_ind)
    # Escolhe o melhor canditado
    best_selected = np.max(selected, axis=1)
    # Gera uma nova população baseado nos candidatos
    P = np.array([pop[best_selected[i]] for i in range(pop_size)])

    return P

def distance_hamming(pop, target):
    pop_size = pop.shape[0]  # Salva o tamanho da população
    # Calcula a quantidade de elementos diferentes
    hamDist = np.array([np.count_nonzero(target != pop[i,]) for i in range(pop_size)])
    return hamDist

# Calcula a aptidão baseada na distancia de Hamming
def hamming_aptitude(s, hamDist):
    apt = np.array([(s - hamDist[i]) for i in range(hamDist.size)])
    return apt

# Converte a bitstring para inteiro
def bitsring_convert(bit_value):
    value = bit_value
    # Transforma em string
    value_string = np.array2string(value,separator='')
    value_string = value_string.lstrip('[').rstrip(']')
    # Converte para inteiro e executa a estrategia de corte
    value_int = int(value_string, 2)
    value_int = cut(value_int,nmin=0,nmax=1000)

    final_value = float(value_int/1000)
    return final_value

# Estrategia de corte
def cut(value_int,nmin,nmax):
    return min(max(value_int,nmin),nmax)

#
def fvalue_aptitude(target_value,pop):
    pop_size = pop.shape[0]
    # Cria o array da função de aptidão
    apt = np.zeros(shape=(pop_size),dtype=float)

    # Converte e calcula a aptidão
    for i in range(pop_size):
        value = bitsring_convert(pop[i, ])

        apt[i] = round((1/(f(target_value) - f(value) + 1)) * 10 , 2)
    return apt

# Função do exercicio 2
def f(x):
    return math.pow(2, -2 * math.pow(((x - 0.1) / 0.9), 2)) * math.pow(math.sin(5 * math.pi * x), 6)

# --------------------------------------------
# Exercicio 3
# --------------------------------------------
# Função Exercicio 3
def func(x,y):
    return ((1 - x) ** 2) + (100 * (y - x ** 2) ** 2)

# Define o bit que utilizado para o valor
def signal_str(value):
    if value < 0:
        signal = '0'
    else:
        signal = '1'
    return signal

# Retorna o sinal positivo ou negativo dependendo do bit
def signal_int(bit):
    if bit == '0':
        signal = -1
    else:
        signal = 1
    return signal
# Converte para bitstring
def convert_bitstring(bit_value):
    value = bit_value
    # Transforma em string
    value_string = np.array2string(value, separator='')
    value_string = value_string.lstrip('[').rstrip(']')
    # Obtem o sinal
    signal = signal_int(value_string[0])
    # Pega o bitstring de x e y e converte para inteiro, executando a estrategia de corte
    value_int = value_string[1:]
    value_int = int(value_int,2) * signal
    value_int = cut(value_int,nmin=-500,nmax=500)

    value_final = float(value_int/100)

    return value_final

def sigf_aptitude(pop):
    pop_size = pop.shape[0]
    # Gera o array de aptidão
    apt = np.zeros(shape=(pop_size), dtype=float)

    # Separa as partes x e y do bitstring e testa na função, obtendo a aptidão
    for i in range(pop_size):
        value_x = convert_bitstring(pop[i,:10])
        value_y = convert_bitstring(pop[i,10:])

        apt[i] = round((1/(func(value_x,value_y) + 1)) * 10,4)

    return apt
# --------------------------------------------
# Funções para gerar graficos e estatisticas
# --------------------------------------------
def stats(bapt_mean, best_apt_value, apt_mean, mean_apt_value, gen_value_total,gen_value,exec_time,gen_time):
    gen_value_total.append(gen_value)
    bapt_mean.append(statistics.mean(best_apt_value))
    apt_mean.append(statistics.mean(mean_apt_value))
    gen_time.append(exec_time)
    return bapt_mean,apt_mean,gen_value_total, gen_time

def write_stats(apt_meanA,gen_numA,gen_timeA,apt_meanB,gen_numB,gen_timeB,file_path,sel_mtd):
    file = open(str(file_path), 'w+')
    file.write(str(sel_mtd)+ ':\n')
    file.write('Media das medias de aptidão por geração: %.2f' % round(statistics.mean(apt_meanA),2) + '\n')
    file.write('Media de gerações necessarias para obter o melhor individuo: %.2f' % round(statistics.mean(gen_numA),2) + '\n')
    file.write('Tempo medio de execução por geração: %4.2f' % round(statistics.mean(gen_timeA),2) + '\n')
    file.write('**************************************************************************' +'\n')
    file.write(str(sel_mtd) + '+Elitismo:\n')
    file.write('Media das medias de aptidão por geração: %.2f' % round(statistics.mean(apt_meanB), 2) + '\n')
    file.write('Media de gerações necessarias para obter o melhor individuo: %.2f' % round(statistics.mean(gen_numB),2) + '\n')
    file.write('Tempo medio de execução por geração: %4.2f' % round(statistics.mean(gen_timeB),2) + '\n')
    file.close()

def write_sucess(sucess_rateA,sucess_rateB,file_path,sel_mtd):
    file = open(str(file_path), 'w+')
    file.write(str(sel_mtd) + ':\n')
    file.write('Taxa de Sucesso: %.2f' % sucess_rateA + '\n')
    file.write('**************************************************************************' + '\n')
    file.write(str(sel_mtd) + '+Elitismo:\n')
    file.write('Taxa de Sucesso: %.2f' % sucess_rateB + '\n')
    file.close()

def plot_gen(gen_numA,gen_numB,file_path,sel_mtd):
    figure_2 = plt.figure()
    plt.plot(gen_numA, 'r', label=str(sel_mtd))
    plt.plot(gen_numB, 'b', label=str(sel_mtd) + 'Elitismo')
    plt.title('Numero de Gerações para atingir o melhor individuo com seleção ' + str(sel_mtd))
    plt.xlabel('Execução')
    plt.ylabel('Numero de Gerações')
    plt.legend()
    figure_2.savefig(str(file_path))
    plt.show()

def plot_time(gen_timeA,gen_timeB,file_path,sel_mtd):
    figure_1 = plt.figure()
    plt.plot(gen_timeA, 'y', label=str(sel_mtd))
    plt.plot(gen_timeB, 'g', label=str(sel_mtd) + 'Elitismo')
    plt.title("Tempo de execução em cada execução com seleção " + str(sel_mtd))
    plt.xlabel('Execução')
    plt.ylabel('Tempo de execução (s)')
    plt.legend()
    figure_1.savefig(str(file_path))
    plt.show()
    
def plot_apt(best_apt_value,mean_apt_value,file_path,sel_mtd):
    figure_3 = plt.figure()
    plt.plot(best_apt_value, 'c', label='Melhor Aptidão da Geração')
    plt.plot(mean_apt_value, 'm', label='Media das Aptidões da Geração')
    plt.title('Aptidão media ' + str(sel_mtd))
    plt.xlabel('Geração')
    plt.ylabel('Aptidão')
    plt.legend()
    figure_3.savefig(str(file_path))
    plt.show()
