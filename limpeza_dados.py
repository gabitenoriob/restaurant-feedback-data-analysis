def limpeza_dados(df):
    """
    Função para limpar e transformar os dados brutos.
    """
    import re
    from textblob import TextBlob

    # Função para analisar o sentimento do comentário
    def analisar_sentimento(comentario):
        if not comentario or not isinstance(comentario, str) or comentario.strip() == "":
            return "neutro"
        analysis = TextBlob(comentario)
        if analysis.sentiment.polarity > 0:
            return "positivo"
        elif analysis.sentiment.polarity < 0:
            return "negativo"
        else:
            return "neutro"

    # Função para calcular a categoria NPS
    def calcula_nps(nota):
        try:
            nota = int(nota)
            if nota >= 9:
                return "promotor"
            elif 7 <= nota <= 8:
                return "passivo"
            elif 0 <= nota <= 6:
                return "detrator"
            else:
                return "desconhecido"
        except (ValueError, TypeError):
            return "desconhecido"

    # Limpeza de colunas de texto
    if 'attendantName' in df and isinstance(df['attendantName'], str):
        df['attendantName'] = df['attendantName'].strip().title()
    else:
        df['attendantName'] = None

    if 'serviceComment' in df and isinstance(df['serviceComment'], str):
        df['serviceComment'] = re.sub(r'\s+', ' ', df['serviceComment']).strip()
    else:
        df['serviceComment'] = None

    # Análise de sentimento
    df['sentimento_servico'] = analisar_sentimento(df.get('serviceComment', None))

    # Cálculo da categoria NPS
    df['categoria_nps'] = calcula_nps(df.get('recommendationRating', None))

    return df