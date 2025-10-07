from datetime import datetime, timezone

import pandas as pd
#

def limpeza_dados(df):
    """
    Função para limpar e transformar os dados brutos.
    """
    import re
    df = df.rename(columns={'id': 'id_feedback_origem'})
    df['data_carga_dw'] = datetime.now(timezone.utc)
    df['data_feedback'] = df['timestamp']
    df = df.drop(columns=['timestamp'])
    
    # Limpeza de comentários: remover espaços extras e normalizar texto
    def clean_text(text):
        if isinstance(text, str):
            text = re.sub(r'\s+', ' ', text)  # Remove espaços extras
            return text.strip()
        return None
    if 'attendant_name' in df and df['attendant_name'].notnull().any() :
        df['attendant_name'] = df['attendant_name'].apply(clean_text)
    else:
        df['attendant_name'] = None

    if 'general_comment' in df and df['general_comment'].notnull().any() :
        df['general_comment'] = df['general_comment'].apply(clean_text)

    return df