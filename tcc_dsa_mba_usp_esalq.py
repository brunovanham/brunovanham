#Bibliotecas
import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


!pip install requests-html --quiet

#Extração dos Dados-Fundamentus FIIs para coleta de Dados
url = 'https://www.fundamentus.com.br/fii_resultado.php'

headers = {
    'User-Agent'      : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    'Accept'          : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language' : 'en-US,en;q=0.5',
    'DNT'             : '1',
    'Connection'      : 'close'
}

data = requests.get(url, headers=headers, timeout=7).text
soup = BeautifulSoup(data, "html.parser")

tabela = soup.find('table')

#Extração dos Dados - Definição Colunas do Data Frame
df = pd.DataFrame(columns=['Papel', 'Segmento', 'FFO Yield %',
                 'Dividend Yield %', 'P/VP', 'Valor de Mercado', 'Liquidez',
                 'Qtd de Imoveis', 'Aluguel por m2', 'Cap Rate %', 'Vacancia Media %'])

#Extração dos Dados - Link valores tabela
for row in tabela.tbody.find_all('tr'):
    #buscando todas as colunas de cada linha
    columns = row.find_all('td')
    if(columns != []):
        papel = columns[0].text.strip(' ')
        segmento = columns[1].text.strip(' ')
        ffo_yield = columns[3].text.strip(' ')
        dividend_yield = columns[4].text.strip(' ')
        p_vp = columns[5].text.strip(' ')
        valor_de_mercado = columns[6].text.strip(' ')
        liquidez = columns[7].text.strip(' ')
        qtd_de_imoveis = columns[8].text.strip(' ')
        preco_do_m2 = columns[9].text.strip(' ')
        aluguel_por_m2 = columns[10].text.strip(' ')
        cap_rate = columns[11].text.strip(' ')
        vacancia_media = columns[12].text.strip(' ')

        df = pd.concat([df, pd.DataFrame.from_records([{'Papel':papel, 'Segmento': segmento,
                                                        'FFO Yield %': ffo_yield, 'Dividend Yield %': dividend_yield,
                                                        'P/VP': p_vp, 'Valor de Mercado': valor_de_mercado,
                                                        'Liquidez': liquidez, 'Qtd de Imoveis': qtd_de_imoveis,
                                                        'Preco do m2': preco_do_m2, 'Aluguel por m2': aluguel_por_m2,
                                                        'Cap Rate %': cap_rate, 'Vacancia Media %': vacancia_media}])],
                                                        ignore_index=True)

df

# Tratamento dos Dados - Removers  milhar
df.iloc[:, 2:] = df.iloc[:, 2:].applymap(lambda x: x.replace('.', ''))

#Tratamento dos Dados - Substituir "%" por ""
df.iloc[:, [2, 3, 9, 10]] = df.iloc[:, [2, 3, 9, 10]].applymap(lambda x: x.replace('%', ''))

#Tratamento dos Dados - Alteracao pontuaçao , por .
df.iloc[:, 2:] = df.iloc[:, 2:].applymap(lambda x: x.replace(',', '.'))

df

#Tratamento dos Dados - Alteracao tipo de dados

df = df.astype({'FFO Yield %': 'float64', 'Dividend Yield %': 'float64', 'P/VP': 'float64', 'Valor de Mercado': 'int64', 'Liquidez': 'int64',
                'Qtd de Imoveis': 'int64', 'Preco do m2': 'float64', 'Aluguel por m2': 'float64', 'Cap Rate %': 'float64', 'Vacancia Media %': 'float64'})
df.info()

df.head()

# prompt: ver tabela completa

df

#Tratamento dos Dados - FILTROS

#FFO Yield %	maior que 0
df = df[df['FFO Yield %'] > 0]

#Dividend Yield % menor que 100
df = df[df['Dividend Yield %'] < 100]

#P/PV maior que 0
df = df[df['P/VP'] > 0.5]

#Valor de Mercado maior que 0
df = df[df['Valor de Mercado'] > 0]

#Liquidez maior que 0
df = df[df['Liquidez'] > 10000]

#Remover Segmento em Branco
df = df[df['Segmento'] != ""]

#Alterando Vacancia Media % Por Ocupacao Total%
df['Ocupacao Total %'] = 100 - df['Vacancia Media %'].astype(float)
df = df.drop('Vacancia Media %', axis=1)
df.head()

df.info()

# Tratamento de Dados - Segmentação

df_titulos_val_mob = df[df['Segmento'] == 'Títulos e Val. Mob.']
df_shoppings = df[df['Segmento'] == 'Shoppings']
df_outros = df[df['Segmento'] == 'Outros']
df_hibrido = df[df['Segmento'] == 'Híbrido']
df_lajes_corporativas = df[df['Segmento'] == 'Lajes Corporativas']
df_logistica = df[df['Segmento'] == 'Logística']
df_residencial = df[df['Segmento'] == 'Residencial']
df_hotel = df[df['Segmento'] == 'Hotel']
df_hospital = df[df['Segmento'] == 'Hospital']

#Tratamento de Dados - Remoção de valores irrelevantes para Fiis de Papel

df_titulos_val_mob = df_titulos_val_mob.drop([ 'Qtd de Imoveis', 'Aluguel por m2', 'Cap Rate %', 'Preco do m2', 'Ocupacao Total %'], axis=1)
df_hibrido = df_hibrido.drop([ 'Qtd de Imoveis', 'Aluguel por m2', 'Cap Rate %', 'Preco do m2', 'Ocupacao Total %'], axis=1)

#Mostrar a quantidade de segmentos

df['Segmento'].value_counts()

# Funcao AHP_G
def AHP_G(seg_original):
    #Monotônico de Custo
    seg_copy = seg_original.copy(deep=True)
    seg_copy['P/VP'] = seg_original['P/VP'].apply(lambda x: 1/x if 'P/VP' in seg_original else x)

    #Normalização dos Critérios da Tabela
    criterios_para_normalizar = ['FFO Yield %', 'Dividend Yield %', 'P/VP', 'Valor de Mercado', 'Liquidez', 'Qtd de Imoveis', 'Aluguel por m2', 'Cap Rate %', 'Preco do m2', 'Ocupacao Total %']
    criterios_presentes = [c for c in criterios_para_normalizar if c in seg_copy.columns]

    if criterios_presentes:
        seg_normalizado = seg_copy[criterios_presentes].apply(lambda x: x / x.sum(), axis=0)


    # Definição do Fator Gaussiano
    fg = seg_normalizado.std() / seg_normalizado.mean()
    # Definição do Fator Normalizado
    fg_normalizado = fg / fg.sum()

    # Ponderação da Matriz de Decisão
    seg_normalizado = seg_normalizado * fg_normalizado

    # Soma das Ponderações da Matriz de Decisão
    seg_normalizado['AHP-G'] = seg_normalizado.sum(axis=1)

    # Ranking AHP-G
    seg_original['AHP-G'] = seg_normalizado['AHP-G']
    seg_original = seg_original.sort_values(by='AHP-G', ascending=False)

    # Adicionar uma coluna "Ranking" no início da tabela
    seg_original.insert(0, "Ranking", range(1, len(seg_original) + 1))

    return seg_original

# Rodar a funcao acima para todos os segmentos

df_titulos_val_mob = AHP_G(df_titulos_val_mob)
df_shoppings = AHP_G(df_shoppings)
df_outros = AHP_G(df_outros)
df_hibrido = AHP_G(df_hibrido)
df_lajes_corporativas = AHP_G(df_lajes_corporativas)
df_logistica = AHP_G(df_logistica)
df_residencial = AHP_G(df_residencial)
df_hotel = AHP_G(df_hotel)
df_hospital = AHP_G(df_hospital)

def stats_AHP_G(resultados_segmento):
    # Monotônico de Custo
    seg_copy = resultados_segmento.copy(deep=True)
    seg_copy['P/VP'] = resultados_segmento['P/VP'].apply(lambda x: 1/x)

    # Normalização dos Valores da Tabela
    criterios_para_normalizar = ['FFO Yield %', 'Dividend Yield %', 'P/VP', 'Valor de Mercado', 'Liquidez', 'Qtd de Imoveis', 'Aluguel por m2', 'Cap Rate %', 'Preco do m2', 'Ocupacao Total %']
    criterios_presentes = [c for c in criterios_para_normalizar if c in seg_copy.columns]

    if criterios_presentes:
        seg_normalizado = seg_copy[criterios_presentes].apply(lambda x: x / x.sum(), axis=0)

    # Estatísticas resumidas
    stats = {
        'Média': seg_normalizado.mean(),
        'Desvio Padrão': seg_normalizado.std(),
        'Fator Gaussiano': seg_normalizado.std() / seg_normalizado.mean(),
        'FG Normalizado': (seg_normalizado.std() / seg_normalizado.mean()) / (seg_normalizado.std() / seg_normalizado.mean()).sum()
    }

    # Criar DataFrame com estatísticas resumidas
    df = pd.DataFrame(stats)

    return df

# Executar Stats para todos os segmentos

df_titulos_val_mob_stats = stats_AHP_G(df_titulos_val_mob)
df_shoppings_stats = stats_AHP_G(df_shoppings)
df_outros_stats = stats_AHP_G(df_outros)
df_hibrido_stats = stats_AHP_G(df_hibrido)
df_lajes_corporativas_stats = stats_AHP_G(df_lajes_corporativas)
df_logistica_stats = stats_AHP_G(df_logistica)
df_residencial_stats = stats_AHP_G(df_residencial)
df_hotel_stats = stats_AHP_G(df_hotel)
df_hospital_stats = stats_AHP_G(df_hospital)

# Criar uma lista com os melhores resultados de cada segmento
melhores_resultados = [
    df_titulos_val_mob.head(1),
    df_shoppings.head(1),
    df_outros.head(1),
    df_hibrido.head(1),
    df_lajes_corporativas.head(1),
    df_logistica.head(1),
    df_residencial.head(1),
    df_hotel.head(1),
    df_hospital.head(1),
]

# Concatenar os melhores resultados em um único DataFrame
df_melhores = pd.concat(melhores_resultados)

# Mover a coluna "AHP-G" para o final da tabela
ultima_coluna = df_melhores.pop("AHP-G")
df_melhores.insert(len(df_melhores.columns), "AHP-G", ultima_coluna)

#Ordenar
df_melhores = df_melhores.sort_values(by="AHP-G", ascending=False)

# Adicionar uma coluna "Ranking" no início da tabela
df_melhores.drop(columns=["Ranking"], inplace=True)
df_melhores.insert(0, "Ranking", range(1, len(df_melhores) + 1))

# Exibir a tabela
df_melhores
