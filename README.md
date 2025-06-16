# MVP Análise de Dados - Lending Club

**Autor:** André Camatta

## Objetivo

Prever se um empréstimo peer-to-peer (P2P) entrará em default no momento da concessão, utilizando dados da Lending Club (2015-2020).

## Dataset

- **Fonte:** [Lending Club 2007-2020Q3](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1)
- **Período:** 2015-2020 (filtrado para regime regulatório homogêneo)
- **Amostra:** `lending_club_sample_2015_2020.csv.gz` (≈400k registros, 53MB)
- **Target:** Classificação binária (default vs. pago)

## Estrutura do Projeto

```
├── mvp_puc.ipynb                      # Notebook principal (executável pelo avaliador)
├── data_processing.py                 # Módulo com pipeline de dados
├── lending_club_sample_2015_2020.csv.gz  # Dataset anonimizado (80MB)
├── LCDataDictionary.xlsx              # Dicionário de dados
├── requirements/dependencies          # pyproject.toml, uv.lock
└── README.md                          # Este arquivo
```

## Reprodução

### Para Avaliadores (Simples)
1. Clone o repositório
2. Instale dependências: `uv sync` ou `pip install pandas numpy`
3. Execute o notebook: `mvp_puc.ipynb`
   - Dataset anonimizado já incluído
   - Não requer configuração de APIs externas

### Para Desenvolvedores (Pipeline Completo)
1. Configure API do Kaggle: coloque `kaggle.json` em `~/.kaggle/`
2. Instale dependências: `uv add kaggle pandas numpy`
3. Use o módulo: `from data_processing import process_lending_club_pipeline`

## Anonimização Aplicada

- ✅ Filtro temporal (2015-2020)
- ✅ Remoção de colunas de vazamento futuro
- ✅ Exclusão de identificadores pessoais (LGPD)
- ✅ Amostragem estratificada (400k registros)
- ✅ Agregação de status em binário (default/pago)

## Tecnologias

- Python 3.12
- Pandas, NumPy
- Jupyter Notebook
- uv (gerenciador de pacotes)