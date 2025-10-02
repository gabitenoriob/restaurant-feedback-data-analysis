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
    

    # Limpeza de colunas de texto
    if 'attendant_name' in df and isinstance(df['attendant_name'], str):
        df['attendant_name'] = df['attendant_name'].strip().title()
    else:
        df['attendant_name'] = None

    if 'service_comment' in df and isinstance(df['service_comment'], str):
        df['service_comment'] = re.sub(r'\s+', ' ', df['service_comment']).strip()
    else:
        df['service_comment'] = None

    if 'food_comment' in df and isinstance(df['food_comment'], str):
        df['food_comment'] = re.sub(r'\s+', ' ', df['food_comment']).strip()
    else:
        df['food_comment'] = None

    return df