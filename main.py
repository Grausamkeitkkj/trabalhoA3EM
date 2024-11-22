import simpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dados = [
    (17,3), (16,6), (16,9), (20,3), (16,6), (11,4), (12,1), (20,3), (23,14), (14,6),
    (10,8), (24,6), (19,3), (24,3), (23,8), (12,7), (16,3), (15,10), (21,2), (14,4),
    (15,6), (15,9), (20,9), (10,4), (21,1), (20,5), (21,11), (15,1), (23,0), (12,4),
    (22,2), (22,3), (23,4), (16,13), (18,6), (15,0), (10,7), (10,3), (16,8), (22,1),
    (16,5), (19,6), (12,5), (17,6), (17,1), (19,6), (14,1), (19,5), (14,0), (12,4),
    (10,9), (18,5), (10,3), (17,4), (21,9), (22,5), (12,4), (18,2), (10,3), (23,8),
    (20,1), (10,4), (21,6), (15,7), (16,3), (16,2), (24,4), (15,3), (14,3), (15,2),
    (20,8), (19,3), (10,6), (15,3), (22,2), (21,3), (21,1), (15,0), (14,6), (14,9),
    (17,5), (11,1), (13,7), (16,5), (17,0), (24,8), (22,6), (23,5), (14,6), (21,8),
    (22,1), (20,8), (14,2), (24,4), (23,4), (18,2), (19,2), (14,3), (17,3), (14,4),
    (12,5), (10,0), (15,8), (17,4), (22,5), (18,3), (17,6), (13,8), (18,9), (17,4),
    (16,1), (23,4), (24,6), (21,5), (12,2), (11,1), (22,1), (23,10), (10,6), (20,8),
    (18,1), (22,6), (15,1), (14,7), (13,9), (18,4), (13,2), (15,3), (11,10), (21,5),
    (18,5), (21,3), (15,2), (15,8), (17,6), (15,4), (24,5), (17,0), (22,5), (19,7),
    (24,6), (11,5), (17,5), (10,9), (16,6), (16,7), (12,1), (24,8), (17,5), (12,4),
    (12,2), (22,2), (20,10), (24,2), (24,3), (13,6), (16,4), (20,0), (12,7), (10,4),
    (24,3), (15,3), (13,7), (14,0), (13,5), (18,7), (21,3), (13,0), (24,5), (18,7),
    (22,7), (10,5), (14,1), (24,6), (19,3), (10,6), (14,0), (12,2), (17,2), (15,5),
    (15,4), (11,6), (21,4), (16,5), (14,6), (19,7), (12,8), (23,3), (12,1), (20,0),
    (19,0), (10,5), (24,5), (19,4), (20,8), (22,0), (11,4), (19,3), (13,11), (13,4),
    (19,8), (18,4), (10,5), (22,8), (23,4), (23,6), (13,7), (11,6), (16,5), (18,9),
    (19,4), (10,1), (11,2), (12,2), (10,13), (19,6), (13,11), (15,1), (14,5), (23,1),
    (23,2), (17,4), (22,6), (16,6), (19,8), (24,6), (21,2), (20,2), (10,5), (19,9),
    (23,7), (14,1), (19,3), (24,2), (17,10), (22,5), (12,4), (11,6), (19,4), (12,8),
    (20,2), (23,6), (23,0), (16,5), (23,5), (10,4), (18,3), (19,4), (17,2), (11,1)
]

# Função de simulação
def atendimento(env, tempo_de_servico, tempo_de_chegada, servidor):
    """Simula o processo de chegada e atendimento de cada cliente."""
    global tempo_ocupado

    yield env.timeout(tempo_de_chegada)  # Tempo até a chegada do cliente
    tempo_chegada = env.now  # Momento de chegada real do cliente
    print(f"\nCliente chegou no tempo {tempo_chegada}. Requer {tempo_de_servico} minutos de serviço.")

    # Solicita o recurso e realiza o atendimento
    with servidor.request() as req:
        yield req
        tempo_inicio_atendimento = env.now
        tempos_espera.append(tempo_inicio_atendimento - tempo_chegada)
        print(f"Cliente sendo atendido no tempo {tempo_inicio_atendimento}.")

        yield env.timeout(tempo_de_servico)  # Tempo de serviço
        tempo_ocupado += tempo_de_servico
        tempo_saida = env.now
        tempos_servico.append(tempo_de_servico)
        tempos_sistema.append(tempo_saida - tempo_chegada)
        print(f"Cliente atendido e saiu no tempo {tempo_saida}.")

# Configuração da simulação
def executar_simulacao(dados):
    global tempo_ocupado

    # Inicializa o ambiente de simulação e o recurso (servidor) com capacidade de atendimento
    env = simpy.Environment()
    capacidade_servidor = 28
    servidor = simpy.Resource(env, capacity=capacidade_servidor)  # Um servidor para atender clientes em fila

    # Cria um processo para cada cliente com base nos dados
    for tempo_servico, tempo_chegada in dados:
        env.process(atendimento(env, tempo_servico, tempo_chegada, servidor))
    
    # Executa a simulação
    env.run()

    # Cálculo das métricas
    total_tempo_simulacao = env.now
    tempo_medio_espera = sum(tempos_espera) / len(tempos_espera) if tempos_espera else 0
    tempo_medio_sistema = sum(tempos_sistema) / len(tempos_sistema) if tempos_sistema else 0
    taxa_utilizacao_servidor = (tempo_ocupado / (total_tempo_simulacao * capacidade_servidor)) if total_tempo_simulacao > 0 else 0

    print("\nMétricas de Desempenho:")
    print(f"Tempo Médio de Espera na Fila: {tempo_medio_espera:.2f} minutos")
    print(f"Tempo Médio no Sistema: {tempo_medio_sistema:.2f} minutos")
    print(f"Taxa de Utilização do Servidor: {taxa_utilizacao_servidor * 100:.2f}%")

    # Gerar gráficos
    gerar_graficos(tempos_espera, tempos_servico, tempos_sistema, taxa_utilizacao_servidor)

# Função para gerar gráficos
def gerar_graficos(tempos_espera, tempos_servico, tempos_sistema, taxa_utilizacao_servidor):
    plt.figure(figsize=(15, 10))

    # Histograma dos tempos de espera
    plt.subplot(2, 2, 1)
    plt.hist(tempos_espera, bins=20, color='blue', edgecolor='black')
    plt.title('Histograma dos Tempos de Espera')
    plt.xlabel('Tempo de Espera (minutos)')
    plt.ylabel('Frequência')

    # Histograma dos tempos de serviço
    plt.subplot(2, 2, 2)
    plt.hist(tempos_servico, bins=20, color='green', edgecolor='black')
    plt.title('Histograma dos Tempos de Serviço')
    plt.xlabel('Tempo de Serviço (minutos)')
    plt.ylabel('Frequência')

    # Histograma dos tempos no sistema
    plt.subplot(2, 2, 3)
    plt.hist(tempos_sistema, bins=20, color='red', edgecolor='black')
    plt.title('Histograma dos Tempos no Sistema')
    plt.xlabel('Tempo no Sistema (minutos)')
    plt.ylabel('Frequência')

    # Gráfico de linha da taxa de utilização do servidor
    plt.subplot(2, 2, 4)
    plt.plot(range(len(tempos_servico)), [taxa_utilizacao_servidor] * len(tempos_servico), color='purple')
    plt.title('Taxa de Utilização do Servidor')
    plt.xlabel('Tempo')
    plt.ylabel('Taxa de Utilização (%)')

    plt.tight_layout()
    plt.show()

# Variáveis globais para armazenar as métricas
tempos_espera = []
tempos_servico = []
tempos_sistema = []
tempo_ocupado = 0  # Para calcular a taxa de utilização do servidor

# Executar simulação
executar_simulacao(dados)

# Transformar os dados em um DataFrame
df = pd.DataFrame(dados, columns=["Tempo de Serviço", "Tempo de Chegada"])

# Calcular a matriz de correlação
correlacao = df.corr()
print("\nMatriz de Correlação:")
print(correlacao)

# Identificar outliers para um conjunto de dados
def calcular_outliers(dados, coluna):
    Q1 = np.percentile(dados[coluna], 25)  # Primeiro quartil
    Q3 = np.percentile(dados[coluna], 75)  # Terceiro quartil
    IQR = Q3 - Q1  # Intervalo interquartil

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # Identificar outliers
    outliers = dados[(dados[coluna] < limite_inferior) | (dados[coluna] > limite_superior)]
    return outliers, limite_inferior, limite_superior

# Cria DataFrame com os dados
df = pd.DataFrame(dados, columns=["Tempo de Serviço", "Tempo de Chegada"])

# Calcula outliers para "Tempo de Serviço"
outliers_servico, limite_inf_servico, limite_sup_servico = calcular_outliers(df, "Tempo de Serviço")
print(f"\nOutliers em 'Tempo de Serviço':\n{outliers_servico}")

# Calcula outliers para "Tempo de Chegada"
outliers_chegada, limite_inf_chegada, limite_sup_chegada = calcular_outliers(df, "Tempo de Chegada")
print(f"Outliers em 'Tempo de Chegada':\n{outliers_chegada}")

print("\nLimites para 'Tempo de Chegada':")
print(f"Limite Inferior: {limite_inf_chegada}, Limite Superior: {limite_sup_chegada}")


plt.figure(figsize=(10, 7))

# Destaca outliers em vermelho
cores = ['red' if (row["Tempo de Serviço"] < limite_inf_servico or
                   row["Tempo de Serviço"] > limite_sup_servico or
                   row["Tempo de Chegada"] < limite_inf_chegada or
                   row["Tempo de Chegada"] > limite_sup_chegada)
         else 'blue' for _, row in df.iterrows()]

plt.scatter(df["Tempo de Chegada"], df["Tempo de Serviço"], c=cores, alpha=0.7, edgecolor='black')
plt.title("Dispersão: Tempo de Chegada x Tempo de Serviço")
plt.xlabel("Tempo de Chegada (minutos)")
plt.ylabel("Tempo de Serviço (minutos)")
plt.grid(True)
plt.show()