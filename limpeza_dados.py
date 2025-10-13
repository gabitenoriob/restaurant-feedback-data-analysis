from datetime import datetime, timezone

import pandas as pd

def limpar_texto(texto):
    """Limpeza leve do texto antes da análise"""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+|https\S+", '', texto)
    texto = re.sub(r"[^a-zA-ZÀ-ÿ\s]", '', texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto
