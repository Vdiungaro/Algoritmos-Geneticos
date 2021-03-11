import numpy as np
import time
import Functions

# ====================================
# Parametros
# ====================================
zero = '111101101111'  # alvo = 0 , representado por bitstring
target = np.array(list(zero), dtype=int)
population_size = 20  # tamanho da população
s = 12  # tamanho do individuo
mc = 0.05  # chance de mutação
cc = 0.70  # chance de crossover

# ===================================
# Algoritmo Genetico
# ===================================
def genetic(pop,target,mc,cc,sel_mtd,elitism):
    opt_ind = False # Variavel que verifica se é o melhor individuo
    mean_apt_values = []  # Lista com a media das aptidões da geração
    best_apt_values = []  # Lista com a melhor aptidão da geração
    num_iter = 0 # Numero de iterações
    pop_size = pop.shape[0]

    while not opt_ind:
        # Calcula a distancia de Hamming para a população
        hamDist = Functions.distance_hamming(pop,target)
        # Calcula a aptidão da população
        apt = Functions.hamming_aptitude(s,hamDist)

        # Caso Elitismo seja selecionado, salva a posição e o melhor individuo da geração atual
        if elitism == True:
            best_ind = np.argmax(apt)
            best_indv = pop[best_ind,]

        # Seleção
        if sel_mtd == 'roulette':
            P = Functions.SelectionbyRoulette(pop,apt)

        if sel_mtd == 'tournament':
            P = Functions.SelectionByTournament(pop,apt,num_selected=3)
        # Reprodução
        P = Functions.reproduction(P,cc,mc,s)

        # Nova Avaliação
        new_hamDist = Functions.distance_hamming(P,target)
        new_apt = Functions.hamming_aptitude(s,new_hamDist)

        # Caso Elitismo seja selecionado, substitui o pior individuo da nova geração pelo melhor da geração antiga
        if elitism == True:
            worst_indv = np.argmin(new_apt)
            P[worst_indv,] = best_indv

        # Salva os valores da população inicial
        if mean_apt_values == []:
            best_apt_values.append(apt[np.argmax(apt)])
            mean_apt_values.append(round(apt.sum()/pop_size))

        # Salva os valores da população atual
        mean_apt_values.append(round(new_apt.sum()/pop_size))
        best_apt_values.append(new_apt[np.argmax(new_apt)])

        #Atualiza a população
        pop = P

        #Verifica se atingiu o valor esperado
        if s in new_apt:
            opt_ind=True

        num_iter += 1

    best_apt_values = np.array(best_apt_values, dtype=int)
    mean_apt_values = np.array(mean_apt_values)

    return pop, num_iter, best_apt_values, mean_apt_values, num_iter+1

# ===================================
def main():
    start = time.time()

    # Variaveis para graficos
    gen_numA, gen_numB, gen_numC, gen_numD = ([] for i in range(4))
    bapt_meanA, bapt_meanB, bapt_meanC, bapt_meanD = ([] for i in range(4))
    gen_timeA, gen_timeB, gen_timeC, gen_timeD = ([] for i in range(4))
    apt_meanA, apt_meanB, apt_meanC, apt_meanD = ([] for i in range(4))
    for i in range(100):
        # Gera a população inicial
        init = np.random.randint(2, size=population_size * s)
        pop = np.reshape(init, (population_size, s))

        #Roleta
        startA = time.time()
        popA, num_iter, best_apt_valueA, mean_apt_valueA, gen_valueA = genetic(pop, target, mc, cc, sel_mtd='roulette',elitism = False)
        endA = time.time()
        exec_time = (endA - startA)
        bapt_meanA, apt_meanA, gen_numA, gen_timeA = Functions.stats(bapt_meanA, best_apt_valueA, apt_meanA, mean_apt_valueA, gen_numA,gen_valueA,exec_time,gen_timeA)

        # Roleta + Elitismo
        startB = time.time()
        popB, num_iter, best_apt_valueB, mean_apt_valueB, gen_valueB = genetic(pop, target, mc, cc, sel_mtd='roulette',elitism=True)
        endB = time.time()
        exec_time = (endB - startB)
        bapt_meanB, apt_meanB, gen_numB, gen_timeB = Functions.stats(bapt_meanB, best_apt_valueB, apt_meanB, mean_apt_valueB, gen_numB, gen_valueB, exec_time, gen_timeB)

        # Torneio
        startC = time.time()
        popC, num_iter, best_apt_valueC, mean_apt_valueC, gen_valueC = genetic(pop, target, mc, cc, sel_mtd='tournament',elitism=False)
        endC = time.time()
        exec_time = (endC - startC)
        bapt_meanC, apt_meanC, gen_numC, gen_timeC = Functions.stats(bapt_meanC, best_apt_valueC, apt_meanC, mean_apt_valueC, gen_numC, gen_valueC, exec_time, gen_timeC)

        # Torneio + Elitismo
        startD = time.time()
        popD, num_iter, best_apt_valueD, mean_apt_valueD, gen_valueD = genetic(pop, target, mc, cc, sel_mtd='tournament',elitism=True)
        endD = time.time()
        exec_time = (endD - startD)
        bapt_meanD, apt_meanD, gen_numD, gen_timeD = Functions.stats(bapt_meanD, best_apt_valueD, apt_meanD,mean_apt_valueD, gen_numD, gen_valueD, exec_time,gen_timeD)

    end = time.time()
    print("tempo de execução:", (end-start), 'seconds')

    # Gera os Graficos de Tempo
    Functions.plot_time(gen_timeA,gen_timeB,file_path="Results/Ex1/tempoRoleta.pdf" ,sel_mtd = 'Roleta')
    Functions.plot_time(gen_timeC,gen_timeD,file_path="Results/Ex1/tempoTorneio.pdf",sel_mtd = 'Torneio')

    # Gera os Graficos de num de gerações
    Functions.plot_gen(gen_numA,gen_numB,file_path="Results/Ex1/geracoesRoleta.pdf",sel_mtd = 'Roleta')
    Functions.plot_gen(gen_numC,gen_numD,file_path="Results/Ex1/geracoesTorneio.pdf",sel_mtd = 'Torneio')

    # Escreve as estatisticas em um arquivo txt
    Functions.write_stats(apt_meanA, gen_numA, gen_timeA, apt_meanB, gen_numB, gen_timeB, file_path='Results/Ex1/estatisticasroleta.txt', sel_mtd='Roleta')
    Functions.write_stats(apt_meanC, gen_numC, gen_timeC, apt_meanD, gen_numD, gen_timeD, file_path='Results/Ex1/estatisticastorneio.txt', sel_mtd='Torneio')

    #Gera os graficos de Aptidão
    Functions.plot_apt(best_apt_valueA, mean_apt_valueA, file_path='Results/Ex1/aptR.pdf',sel_mtd='Roleta')
    Functions.plot_apt(best_apt_valueB, mean_apt_valueB, file_path='Results/Ex1/aptRE.pdf',sel_mtd='Roleta Com Elitismo')
    Functions.plot_apt(best_apt_valueC, mean_apt_valueC, file_path='Results/Ex1/aptT.pdf', sel_mtd='Torneio')
    Functions.plot_apt(best_apt_valueD, mean_apt_valueD, file_path='Results/Ex1/aptTE.pdf', sel_mtd='Torneio Com Elitismo')
if __name__ == '__main__':
        main()


