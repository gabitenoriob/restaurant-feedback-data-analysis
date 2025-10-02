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

    if 'service_comment' in df and df['service_comment'].notnull().any():
        df['service_comment'] = df['service_comment'].apply(clean_text)
    else:
        df['service_comment'] = None

    if 'food_comment' in df and df['food_comment'].notnull().any():
        df['food_comment'] = df['food_comment'].apply(clean_text)
    else:
        df['food_comment'] = None
    if 'enviroment_comment' in df and df['enviroment_comment'].notnull().any():
        df['enviroment_comment'] = df['enviroment_comment'].apply(clean_text)
    else:
        df['enviroment_comment'] = None

    return df