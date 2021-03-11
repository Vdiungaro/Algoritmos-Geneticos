import numpy as np
import time
import Functions

# ====================================
# Parametros
# ====================================
target_x = 1  # Valor alvo que minimiza a função no valor x
target_y = 1  # Valor alvo que minimiza a função no valor y
population_size = 80  # tamanho da população
s = 20  # tamanho do individuo
mc = 0.15  # chance mutação
cc = 0.80  # chance crossover
# ===================================
# Algoritmo Genetico
# ===================================

def genetic(pop,target,mc,cc,sel_mtd,elitism):
    opt_ind = False  # Variavel que verifica se é o melhor individuo
    mean_apt_values = []  # Lista com a media das aptidões da geração
    best_apt_values = []  # Lista com a melhor aptidão da geração
    obj = None  # Variavel que verifica se foi um sucesso.

    num_iter = 0  # Numero de iterações

    pop_size = pop.shape[0]
    ind_size = pop.shape[1]

    stop_val = 0  # Contador que limita o numero maximo de gerações
    err_val = 0.15  # Valor que a aptidão deve melhorar

    while not opt_ind:
        # Calcula a aptidão da população
        apt = Functions.sigf_aptitude(pop)

        #Caso Elitismo seja selecionado, salva a posição e o melhor individuo da geração atual
        if elitism == True:
            best_ind = np.argmax(apt)
            best_indv = pop[best_ind,]

        # Seleção
        if sel_mtd == 'roulette':
            P = Functions.SelectionbyRoulette(pop,apt)

        if sel_mtd == 'tournament':
            P = Functions.SelectionByTournament(pop,apt,num_selected=3)

        # Reprodução
        P = Functions.reproduction(P,cc,mc,ind_size)

        # Nova Avaliação
        new_apt = Functions.sigf_aptitude(P)

        # Caso Elitismo seja selecionado, substitui o pior individuo da nova geração pelo melhor da geração antiga
        if elitism == True:
            worst_indv = np.argmin(new_apt)
            P[worst_indv,] = best_indv

        # Salva os valores da população inicial
        if best_apt_values == []:
            best_apt_values.append(apt[np.argmax(apt)])
            mean_apt_values.append(round(apt.sum() / pop_size))

        # Salva os valores da população atual
        best_apt_values.append(new_apt[np.argmax(new_apt)])
        mean_apt_values.append(round(new_apt.sum() / pop_size))

        #Atualiza a população
        pop = P

        # Obtem o valor da aptidão maxima
        max_val = np.amax(new_apt)

        #Verifica se atingiu o valor esperado e atribui um sucesso
        if 10.0 in new_apt:
            #print('SUCESSO')
            obj = 'Sucesso'
            opt_ind = True

        # Verifica se a aptidão melhorou
        if max_val <= max_val+err_val:
            stop_val += 1
        # Caso a aptidão não melhore por 350 gerações, a execução é terminada
        if stop_val == 350:
            #print('FALHOU')
            opt_ind =True

        num_iter += 1

    best_apt_values = np.array(best_apt_values)
    mean_apt_values = np.array(mean_apt_values)

    return pop, num_iter, best_apt_values, mean_apt_values, num_iter+1, obj

# ===================================
def main():
    start = time.time()

    global target_x
    global target_y

    # Converte a parte x do valor alvo para bitstring, executando a estrategia de corte
    target_x = Functions.cut(target_x * 100, nmin=-500,nmax=500)
    target_x = Functions.signal_str(target_x) + format(int(abs(target_x)), '09b')
    target_x = np.array(list(target_x), dtype=int)

    # Converte a parte y do valor alvo para bitstring, executando a estrategia de corte
    target_y = Functions.cut(target_y * 100, nmin=-500, nmax=500)
    target_y = Functions.signal_str(target_y) + format(int(abs(target_y)), '09b')
    target_y = np.array(list(target_y), dtype=int)

    # Junta os alvos em um bitstring unico [target_x;target_y]
    target = np.concatenate((target_x,target_y), axis=None)

    # Variaveis para graficos
    gen_numA, gen_numB, gen_numC, gen_numD = ([] for i in range(4))
    bapt_meanA, bapt_meanB, bapt_meanC, bapt_meanD = ([] for i in range(4))
    gen_timeA, gen_timeB, gen_timeC, gen_timeD = ([] for i in range(4))
    apt_meanA, apt_meanB, apt_meanC, apt_meanD = ([] for i in range(4))
    sucess_rateA = 0  # contador de succesoA
    sucess_rateB = 0  # contador de succesoB
    sucess_rateC = 0  # contador de succesoC
    sucess_rateD = 0  ## contador de succesoD
    for i in range(100):
        # Gera a população inicial
        init = np.random.randint(2, size=population_size * s)
        pop = np.reshape(init, (population_size, s))

        # Roleta
        startA = time.time()
        popA, num_iter, best_apt_valueA, mean_apt_valueA, gen_valueA, objA = genetic(pop, target, mc, cc, sel_mtd='roulette',elitism=False)
        endA = time.time()
        exec_time = (endA - startA)
        if objA == 'Sucesso':  # Verifica se foi um sucesso e atribui +1 valor ao contador
            sucess_rateA += 1
        bapt_meanA, apt_meanA, gen_numA, gen_timeA = Functions.stats(bapt_meanA, best_apt_valueA, apt_meanA,mean_apt_valueA, gen_numA, gen_valueA, exec_time,gen_timeA)

        # Roleta + Elitismo
        startB = time.time()
        popB, num_iter, best_apt_valueB, mean_apt_valueB, gen_valueB, objB = genetic(pop, target, mc, cc, sel_mtd='roulette',elitism=True)
        endB = time.time()
        exec_time = (endB - startB)
        if objB == 'Sucesso':  # Verifica se foi um sucesso e atribui +1 valor ao contador
            sucess_rateB += 1
        bapt_meanB, apt_meanB, gen_numB, gen_timeB = Functions.stats(bapt_meanB, best_apt_valueB, apt_meanB,mean_apt_valueB, gen_numB, gen_valueB, exec_time,gen_timeB)

        # Torneio
        startC = time.time()
        popC, num_iter, best_apt_valueC, mean_apt_valueC, gen_valueC, objC = genetic(pop, target, mc, cc,sel_mtd='tournament', elitism=False)
        endC = time.time()
        exec_time = (endC - startC)
        if objC == 'Sucesso':  # Verifica se foi um sucesso e atribui +1 valor ao contador
            sucess_rateC += 1
        bapt_meanC, apt_meanC, gen_numC, gen_timeC = Functions.stats(bapt_meanC, best_apt_valueC, apt_meanC,mean_apt_valueC, gen_numC, gen_valueC, exec_time,gen_timeC)

        # Torneio + Elitismo
        startD = time.time()
        popD, num_iter, best_apt_valueD, mean_apt_valueD, gen_valueD, objD = genetic(pop, target, mc, cc,sel_mtd='tournament', elitism=True)
        endD = time.time()
        exec_time = (endD - startD)
        if objD == 'Sucesso':  # Verifica se foi um sucesso e atribui +1 valor ao contador
            sucess_rateD += 1
        bapt_meanD, apt_meanD, gen_numD, gen_timeD = Functions.stats(bapt_meanD, best_apt_valueD, apt_meanD,mean_apt_valueD, gen_numD, gen_valueD, exec_time,gen_timeD)

    end = time.time()
    print("tempo de execução:", (end - start), 'seconds')
    #  print(sucess_rateA,sucess_rateB,sucess_rateC,sucess_rateD)
    # Gera os Graficos de Tempo
    Functions.plot_time(gen_timeA, gen_timeB, file_path="Results/Ex3/tempoRoleta.pdf", sel_mtd='Roleta')
    Functions.plot_time(gen_timeC, gen_timeD, file_path="Results/Ex3/tempoTorneio.pdf", sel_mtd='Torneio')

    # Gera os Graficos de num de gerações
    Functions.plot_gen(gen_numA, gen_numB, file_path="Results/Ex3/geracoesRoleta.pdf", sel_mtd='Roleta')
    Functions.plot_gen(gen_numC, gen_numD, file_path="Results/Ex3/geracoesTorneio.pdf", sel_mtd='Torneio')

    # Escreve as estatisticas em um arquivo txt
    Functions.write_stats(apt_meanA, gen_numA, gen_timeA, apt_meanB, gen_numB, gen_timeB,file_path='Results/Ex3/estatisticasroleta.txt', sel_mtd='Roleta')
    Functions.write_stats(apt_meanC, gen_numC, gen_timeC, apt_meanD, gen_numD, gen_timeD,file_path='Results/Ex3/estatisticastorneio.txt', sel_mtd='Torneio')

    #Escreve as estatisticas de sucesso em um arquivo txt
    Functions.write_sucess(sucess_rateA,sucess_rateB,file_path='Results/Ex3/sucessoroleta.txt', sel_mtd='Roleta')
    Functions.write_sucess(sucess_rateC, sucess_rateD, file_path='Results/Ex3/sucessotorneio.txt', sel_mtd='Torneio')

    # Gera os graficos de Aptidão
    Functions.plot_apt(best_apt_valueA, mean_apt_valueA, file_path='Results/Ex3/aptR.pdf', sel_mtd='Roleta')
    Functions.plot_apt(best_apt_valueB, mean_apt_valueB, file_path='Results/Ex3/aptRE.pdf',sel_mtd='Roleta Com Elitismo')
    Functions.plot_apt(best_apt_valueC, mean_apt_valueC, file_path='Results/Ex3/aptT.pdf', sel_mtd='Torneio')
    Functions.plot_apt(best_apt_valueD, mean_apt_valueD, file_path='Results/Ex3/aptTE.pdf',sel_mtd='Torneio Com Elitismo')

if __name__ == '__main__':
        main()