# MVP: Previsão de Inadimplência na Lending Club

**Autor:** André Camatta

## 1. Visão Geral

Este projeto é um MVP de ciência de dados que aborda o problema de **previsão de inadimplência (*default*)** em empréstimos *peer-to-peer* (P2P), utilizando um dataset público da **Lending Club**. O objetivo é desenvolver um modelo de classificação binária para estimar a probabilidade de um empréstimo não ser pago, servindo como uma ferramenta de suporte à decisão para investidores em plataformas semelhantes.

O estudo utiliza uma amostra de **600.000 empréstimos** concedidos entre **2015 e 2020**, preparada para garantir representatividade e conformidade com as boas práticas de desenvolvimento.

## 2. Metodologia

O projeto foi estruturado em três etapas principais:

1.  **Análise Exploratória de Dados (EDA)**: Investigação inicial dos dados para extrair insights, compreender a relação entre as variáveis e guiar a etapa de pré-processamento.

2.  **Pipeline de Pré-processamento**: Construção de um pipeline para preparar os dados para a modelagem, incluindo seleção de features, imputação de valores ausentes e transformação de variáveis numéricas (logarítmica, padronização) e categóricas (ordinal, one-hot e target encoding).

3.  **Modelagem Preditiva**: Utilização do algoritmo **LightGBM** para a classificação, com tratamento para o desbalanceamento de classes e validação através de **validação cruzada estratificada**, garantindo a robustez do modelo.

## 3. Estrutura do Repositório

```
.
├── dataset_anonymization.py    # Script para anonimizar e amostrar o dataset.
├── lending_club_sample_2015_2020.csv.gz # Amostra de dados utilizada.
├── mvp_puc_lending_club.ipynb  # Notebook Jupyter com toda a análise.
├── pyproject.toml              # Dependências do projeto.
└── README.md                   # Este arquivo.
```

## 4. Como Reproduzir a Análise

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/andrecamatta/mvp_puc_analise_dados.git
    cd mvp_puc_analise_dados
    ```

2.  **Instale as dependências:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn lightgbm geopandas
    ```

3.  **Execute o Notebook:**
    Abra e execute o notebook `mvp_puc_lending_club.ipynb` em um ambiente Jupyter (Jupyter Lab, VS Code etc.).
