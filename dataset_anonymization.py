"""
Módulo para Anonimização e Amostragem Estratificada - Lending Club Dataset

Este módulo contém as funções especializadas para:
1. Download automático de dados do Kaggle
2. Anonimização completa (LGPD-compliant)
3. Amostragem estratificada por ano e target
4. Preparação de dataset para GitHub (<100MB)

Objetivo: Preparar amostra representativa para análise reprodutível
Autor: André Camatta
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List


def download_kaggle_dataset(dataset_id: str, download_path: str = '.') -> str:
    """
    Baixa dataset do Kaggle usando a API.
    
    Requer configuração prévia do kaggle.json em ~/.kaggle/
    
    Args:
        dataset_id: ID do dataset no Kaggle (formato: usuario/dataset)
        download_path: Diretório onde salvar os arquivos
        
    Returns:
        Nome do arquivo principal baixado
        
    Raises:
        ImportError: Se kaggle não estiver instalado
        Exception: Se houver erro no download
    """
    try:
        import kaggle
    except ImportError:
        raise ImportError("Biblioteca 'kaggle' não encontrada. Instale com: uv add kaggle")
    
    print(f"Baixando dataset: {dataset_id}")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_id, path=download_path, unzip=True)
    
    # Procurar arquivo principal (assume que é o maior arquivo CSV/gzip)
    files = os.listdir(download_path)
    data_files = [f for f in files if f.endswith(('.csv', '.gzip', '.csv.gz'))]
    
    if not data_files:
        raise Exception("Nenhum arquivo de dados encontrado após download")
    
    # Retorna o maior arquivo (provavelmente o principal)
    main_file = max(data_files, key=lambda f: os.path.getsize(os.path.join(download_path, f)))
    print(f"Dataset baixado: {main_file}")
    
    return main_file


def load_lending_club_data(file_path: str) -> pd.DataFrame:
    """
    Carrega dados do Lending Club detectando automaticamente o formato.
    
    Args:
        file_path: Caminho para o arquivo de dados
        
    Returns:
        DataFrame com os dados carregados
    """
    print(f"Carregando dados de: {file_path}")
    
    # Detectar se é comprimido ou não
    if file_path.endswith('.gz'):
        compression = 'gzip'
    elif file_path.endswith('.gzip'):
        # Verificar se realmente é gzip ou apenas extensão enganosa
        with open(file_path, 'rb') as f:
            magic = f.read(2)
            compression = 'gzip' if magic == b'\x1f\x8b' else None
    else:
        compression = None
    
    df = pd.read_csv(file_path, compression=compression, low_memory=False)
    print(f"Dados carregados: {df.shape}")
    
    return df


def anonymize_and_filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica anonimização e filtragem conforme especificações do projeto.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame filtrado e anonimizado
    """
    print("Iniciando anonimização e filtragem...")
    
    # 1. Filtrar período 2015-2020
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    df_filtered = df[(df['issue_d'] >= '2015-01-01') & (df['issue_d'] <= '2020-12-31')].copy()
    print(f"   Após filtro temporal (2015-2020): {df_filtered.shape}")
    
    # 2. Criar variável target binária
    paid_status = ['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid']
    default_status = ['Charged Off', 'Default', 'Late (31-120 days)', 'In Grace Period', 
                     'Does not meet the credit policy. Status:Charged Off', 'Late (16-30 days)']
    
    df_filtered = df_filtered[df_filtered['loan_status'].isin(paid_status + default_status)]
    df_filtered['target_default'] = df_filtered['loan_status'].isin(default_status).astype(int)
    
    target_dist = df_filtered['target_default'].value_counts(normalize=True)
    print(f"   Distribuição target: {target_dist[0]:.1%} pagos, {target_dist[1]:.1%} default")
    
    # 3. Remover colunas com vazamento de informação futura
    future_leak_cols = ['last_pymnt_d', 'total_pymnt', 'recoveries', 'collection_recovery_fee', 
                       'last_credit_pull_d', 'out_prncp', 'out_prncp_inv', 'total_pymnt_inv',
                       'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'hardship_flag',
                       'settlement_status', 'settlement_date', 'settlement_amount', 'debt_settlement_flag']
    
    existing_leak_cols = [col for col in future_leak_cols if col in df_filtered.columns]
    df_filtered = df_filtered.drop(columns=existing_leak_cols)
    print(f"   Removidas {len(existing_leak_cols)} colunas de vazamento")
    
    # 4. Remover/Anonimizar identificadores diretos (LGPD)
    privacy_cols = ['member_id', 'emp_title', 'url', 'desc', 'title']
    existing_privacy_cols = [col for col in privacy_cols if col in df_filtered.columns]
    df_filtered = df_filtered.drop(columns=existing_privacy_cols)
    print(f"   Removidas {len(existing_privacy_cols)} colunas de privacidade")
    
    print(f"Anonimização concluída: {df_filtered.shape}")
    return df_filtered


def create_stratified_sample(df: pd.DataFrame, target_size: int = 600000, 
                           random_state: int = 42) -> pd.DataFrame:
    """
    Cria amostra estratificada por ano e target, mantendo representatividade.
    
    Args:
        df: DataFrame filtrado
        target_size: Tamanho desejado da amostra
        random_state: Seed para reprodutibilidade
        
    Returns:
        DataFrame com amostra estratificada
    """
    print(f"Criando amostra estratificada de {target_size:,} registros...")
    
    # Criar estratos por ano e target
    df['ano'] = df['issue_d'].dt.year
    df['strata'] = df['ano'].astype(str) + '_' + df['target_default'].astype(str)
    
    # Calcular proporções originais
    strata_counts = df['strata'].value_counts()
    strata_props = strata_counts / len(df)
    
    print(f"   Estratos identificados: {len(strata_counts)}")
    
    # Calcular tamanho de cada estrato na amostra
    sample_sizes = {}
    for strata, prop in strata_props.items():
        sample_size = int(target_size * prop)
        if sample_size == 0 and strata_counts[strata] > 0:
            sample_size = 1  # Garantir pelo menos 1 por estrato
        sample_sizes[strata] = min(sample_size, strata_counts[strata])
    
    # Amostragem estratificada
    dfs_sample = []
    for strata, sample_size in sample_sizes.items():
        if sample_size > 0:
            strata_data = df[df['strata'] == strata]
            if len(strata_data) >= sample_size:
                sampled = strata_data.sample(n=sample_size, random_state=random_state)
            else:
                sampled = strata_data
            dfs_sample.append(sampled)
    
    # Combinar amostras
    df_sample = pd.concat(dfs_sample, ignore_index=True)
    
    # Limpar colunas auxiliares
    df_sample = df_sample.drop(['strata'], axis=1)
    
    # Verificar representatividade
    original_year_dist = df['ano'].value_counts(normalize=True).sort_index()
    sample_year_dist = df_sample['ano'].value_counts(normalize=True).sort_index()
    max_diff = abs(original_year_dist - sample_year_dist).max()
    
    print(f"Amostra criada: {len(df_sample):,} registros")
    print(f"   Máxima diferença temporal: {max_diff:.1%}")
    
    return df_sample


def save_sample_for_github(df: pd.DataFrame, filename: str = 'lending_club_sample_2015_2020.csv.gz') -> Dict[str, float]:
    """
    Salva amostra comprimida para upload no GitHub.
    
    Args:
        df: DataFrame da amostra
        filename: Nome do arquivo a salvar
        
    Returns:
        Dicionário com informações do arquivo (tamanho, adequação para GitHub)
    """
    print(f"Salvando amostra: {filename}")
    
    df.to_csv(filename, index=False, compression='gzip')
    
    file_size_mb = os.path.getsize(filename) / (1024**2)
    is_github_suitable = file_size_mb <= 100
    
    info = {
        'file_size_mb': file_size_mb,
        'records': len(df),
        'github_suitable': is_github_suitable
    }
    
    status = "OK" if is_github_suitable else "GRANDE"
    print(f"{status} Arquivo salvo: {file_size_mb:.1f} MB, {len(df):,} registros")
    
    return info


def analyze_temporal_distribution(df_original: pd.DataFrame, df_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Analisa e compara distribuição temporal entre dataset original e amostra.
    
    Args:
        df_original: DataFrame original filtrado
        df_sample: DataFrame da amostra
        
    Returns:
        DataFrame com comparação das distribuições
    """
    df_original['ano'] = df_original['issue_d'].dt.year
    df_sample['ano'] = df_sample['issue_d'].dt.year
    
    original_dist = df_original['ano'].value_counts(normalize=True).sort_index()
    sample_dist = df_sample['ano'].value_counts(normalize=True).sort_index()
    
    comparison = pd.DataFrame({
        'Original (%)': (original_dist * 100).round(1),
        'Amostra (%)': (sample_dist * 100).round(1),
        'Diferença (pp)': abs((original_dist - sample_dist) * 100).round(1)
    })
    
    return comparison


def process_lending_club_pipeline(dataset_id: str = 'ethon0426/lending-club-20072020q1',
                                target_sample_size: int = 600000) -> Tuple[pd.DataFrame, Dict]:
    """
    Pipeline completo de processamento dos dados do Lending Club.
    
    Args:
        dataset_id: ID do dataset no Kaggle
        target_sample_size: Tamanho da amostra final
        
    Returns:
        Tupla com (DataFrame da amostra, informações do processo)
    """
    try:
        # 1. Download
        main_file = download_kaggle_dataset(dataset_id)
        
        # 2. Carregamento
        df = load_lending_club_data(main_file)
        
        # 3. Anonimização e filtragem
        df_filtered = anonymize_and_filter_data(df)
        
        # 4. Amostragem estratificada
        df_sample = create_stratified_sample(df_filtered, target_sample_size)
        
        # 5. Salvamento
        file_info = save_sample_for_github(df_sample)
        
        # 6. Análise temporal
        temporal_analysis = analyze_temporal_distribution(df_filtered, df_sample)
        
        process_info = {
            'original_shape': df.shape,
            'filtered_shape': df_filtered.shape,
            'sample_shape': df_sample.shape,
            'file_info': file_info,
            'temporal_analysis': temporal_analysis
        }
        
        print("\nPipeline completo executado com sucesso!")
        
        return df_sample, process_info
        
    except Exception as e:
        print(f"Erro no pipeline: {e}")
        raise
