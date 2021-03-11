# Vitor Diungaro - PSO

import numpy as np
import random
import time
import Functions

# f(x, y) = (1 − x)^2 + 100(y − x^2)^2
# tem mínimo global f(x,y) = 0 em x = 1 e y = 1
def func(x,y):
    return ((1-x)**2) + (100*(y-(x**2))**2)

# aptidão
def apt(particle):
    return round(func(particle[0],particle[1]),4)

# PSO
def pso(pop_size,max_it,ac_1,ac_2,v_max,v_min, min_stop):

    # Armzena valores medio e minimo de f(x,y)
    fs_min = []
    fs_med = []

    # Inicia o Enxame
    X = np.random.uniform(-5,+5,size=(pop_size*2))
    X = np.reshape(X,(pop_size,2))

    # Inicia a Velocidade
    V = np.random.uniform(v_min,v_max,size=(pop_size*2))
    V = np.reshape(V,(pop_size,2))

    # Armazena a aptidão do individuo
    best_apt = np.array([[X[i],apt(X[i])] for i in range(pop_size)])

    # Cria vizinhança
    neighborhood = np.array([[i,i,[i-1,i+1]] for i in range(pop_size)])
    neighborhood[:,2][pop_size-1][1] = 0 # caso o vizinho da direita seja o ultimo elemento, utiliza o primeiro elemento

    for t in range(max_it):
        fs_min.append(best_apt[best_apt[:,1].argmin(),1])
        fs_med.append(np.mean(best_apt[:,1], dtype=np.float64))

        for i in range(pop_size):
            # Melhor desempenho de cada individuo
            current_apt = apt(X[i])
            if current_apt < best_apt[i,1]:
                best_apt[i] = [X[i], current_apt]

            # Vizinhança
            for j in neighborhood[i,2]:
                current_apt_neighborhood = apt(X[j])
                if current_apt_neighborhood < best_apt[neighborhood[i,1],1]:
                    neighborhood[i,1] = j

        # Atualiza Velocidades
        V = V + (np.random.uniform(0, ac_1, pop_size * 2).reshape((pop_size, 2)) * (best_apt[i, 0] - X[i])) + (
                    np.random.uniform(0, ac_2, pop_size * 2).reshape((pop_size, 2)) * (
                        best_apt[neighborhood[i, 1], 0] - X[i]))
        V = np.clip (V,v_min,v_max)

        # Atualiza a posição dos individuos
        X = X + V
        X = np.clip(X,-5,5)

        # Para a execução se encontrar o minimo global
        if min_stop and (0.0 in fs_min):
            break

    # Retorna o indv mais apto e os valores minimos e medios de f(x,y)
    best = best_apt[:,1].argmin()
    return best_apt[best,0], fs_min, fs_med



def main():
    start = time.time()

    max_it = 250
    pop_size = [(i+1)*10 for i in range(5)]

    # Variaveis para os graficos
    exec_timeA, exec_timeB, exec_timeC, exec_timeD, exec_timeE = ([] for i in range(5))
    num_iter_totalA, num_iter_totalB, num_iter_totalC, num_iter_totalD, num_iter_totalE = ([] for i in range(5))
    fs_min_meanA, fs_min_meanB, fs_min_meanC, fs_min_meanD, fs_min_meanE = ([] for i in range(5))
    fs_med_meanA, fs_med_meanB, fs_med_meanC, fs_med_meanD, fs_med_meanE = ([] for i in range(5))

    for i in range(100):
        startA = time.time()
        PA, fs_minA, fs_medA = pso(pop_size[0], max_it, ac_1=2.05, ac_2=2.05, v_max=2, v_min=-2,
                                           min_stop=True)
        endA = time.time()

        # n_particulas = 20
        startB = time.time()
        PB, fs_minB, fs_medB = pso(pop_size[1], max_it, ac_1=2.05, ac_2=2.05, v_max=2, v_min=-2,
                                           min_stop=True)
        endB = time.time()

        # n_particulas = 30
        startC = time.time()
        PC, fs_minC, fs_medC = pso(pop_size[2], max_it, ac_1=2.05, ac_2=2.05, v_max=2, v_min=-2,
                                           min_stop=True)
        endC = time.time()

        # n_particulas = 40
        startD = time.time()
        PD, fs_minD, fs_medD = pso(pop_size[3], max_it, ac_1=2.05, ac_2=2.05, v_max=2, v_min=-2,
                                           min_stop=True)
        endD = time.time()

        # n_particulas = 50
        startE = time.time()
        PE, fs_minE, fs_medE = pso(pop_size[4], max_it, ac_1=2.05, ac_2=2.05, v_max=2, v_min=-2,
                                           min_stop=True)
        endE = time.time()

        if i == 50:
            Functions.plot_fs(fs_minA, fs_medA, pop_size[0], file_path="ResultsPso/exec50_10.pdf")
            Functions.plot_fs(fs_minB, fs_medB, pop_size[1], file_path="ResultsPso/exec50_20.pdf")
            Functions.plot_fs(fs_minC, fs_medC, pop_size[2], file_path="ResultsPso/exec50_30.pdf")
            Functions.plot_fs(fs_minD, fs_medD, pop_size[3], file_path="ResultsPso/exec50_40.pdf")
            Functions.plot_fs(fs_minE, fs_medE, pop_size[4], file_path="ResultsPso/exec50_50.pdf")

        fs_min_meanA, fs_med_meanA, num_iter_totalA, exec_timeA = Functions.append_values(fs_min_meanA, fs_med_meanA,
                                                                                          fs_minA, fs_medA,
                                                                                          num_iter_totalA, len(fs_minA),
                                                                                          exec_timeA, startA, endA)
        fs_min_meanB, fs_med_meanB, num_iter_totalB, exec_timeB = Functions.append_values(fs_min_meanB, fs_med_meanB,
                                                                                          fs_minB, fs_medB,
                                                                                          num_iter_totalB, len(fs_minB),
                                                                                          exec_timeB, startB, endB)
        fs_min_meanC, fs_med_meanC, num_iter_totalC, exec_timeC = Functions.append_values(fs_min_meanC, fs_med_meanC,
                                                                                          fs_minC, fs_medC,
                                                                                          num_iter_totalC, len(fs_minC),
                                                                                          exec_timeC, startC, endC)
        fs_min_meanD, fs_med_meanD, num_iter_totalD, exec_timeD = Functions.append_values(fs_min_meanD, fs_med_meanD,
                                                                                          fs_minD, fs_medD,
                                                                                          num_iter_totalD, len(fs_minD),
                                                                                          exec_timeD, startD, endD)
        fs_min_meanE, fs_med_meanE, num_iter_totalE, exec_timeE = Functions.append_values(fs_min_meanE, fs_med_meanE,
                                                                                          fs_minE, fs_medE,
                                                                                          num_iter_totalE, len(fs_minE),
                                                                                          exec_timeE, startE, endE)

    num_iters = Functions.append_num_iters(num_iter_totalA, num_iter_totalB, num_iter_totalC, num_iter_totalD,
                                           num_iter_totalE)
    exec_times = Functions.append_exec_times(exec_timeA, exec_timeB, exec_timeC, exec_timeD, exec_timeE)

    # Estatisticas
    # n_particulas = 10
    Functions.write_statistics(pop_size[0], fs_med_meanA, num_iters[0], exec_times[0],
                               file_path="ResultsPso/estatisticasPSO.txt")
    # n_particulas = 20
    Functions.write_statistics(pop_size[1], fs_med_meanB, num_iters[1], exec_times[1],
                               file_path="ResultsPso/estatisticasPSO.txt")
    # n_particulas = 30
    Functions.write_statistics(pop_size[2], fs_med_meanC, num_iters[2], exec_times[2],
                               file_path="ResultsPso/estatisticasPSO.txt")
    # n_particulas = 40
    Functions.write_statistics(pop_size[3], fs_med_meanD, num_iters[3], exec_times[3],
                               file_path="ResultsPso/estatisticasPSO.txt")
    # n_particulas = 50
    Functions.write_statistics(pop_size[4], fs_med_meanE, num_iters[4], exec_times[4],
                               file_path="ResultsPso/estatisticasPSO.txt")

    # Gráficos de média de iterações para encontrar minimo de f(x,y)
    Functions.plot_fxy_over_time(num_iters, pop_size, file_path="ResultsPso/geracoesPorExecucaoPSO.pdf")

    # Grafíco de tempo para encontrar minimo de f(x,y)
    # exec_times = functions.append_exec_times(exec_timeA[:25],exec_timeB[:25],exec_timeC[:25],exec_timeD[:25],exec_timeE[:25])
    Functions.plot_execution_time_over_time(exec_times, pop_size,
                                            file_path="ResultsPso/tempoPorExecucaoPSO.pdf")

    end = time.time()
    print("execution time:", round(end - start, 4), "seconds.")


if __name__ == '__main__':
    main()
