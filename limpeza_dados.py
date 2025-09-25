from datetime import datetime, timezone

import pandas as pd


def limpeza_dados(df):
    """
    Função para limpar e transformar os dados brutos.
    """
    import re
    df = df.rename(columns={'id': 'id_feedback_origem'})
    df = df.drop(columns=['id'])
    df['data_carga_dw'] = datetime.now(timezone.utc)
    df['data_feedback'] = df['timestamp'].dt.date
    



    # Limpeza de colunas de texto
    if 'attendantName' in df and isinstance(df['attendantName'], str):
        df['attendantName'] = df['attendantName'].strip().title()
    else:
        df['attendantName'] = None

    if 'serviceComment' in df and isinstance(df['serviceComment'], str):
        df['serviceComment'] = re.sub(r'\s+', ' ', df['serviceComment']).strip()
    else:
        df['serviceComment'] = None

    return df