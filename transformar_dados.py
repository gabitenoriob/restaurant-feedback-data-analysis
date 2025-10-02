# Função para analisar o sentimento do comentário
from textblob import TextBlob

def transformar_dados(df):
    """
    Função para transformar os dados limpos, adicionando colunas de análise de sentimento e categoria NPS.
    """
    # Análise de sentimento para comentários de serviço, comida e ambiente
    df['sentimento_servico'] = df['service_comment'].apply(analisar_sentimento)
    df['sentimento_comida'] = df['food_comment'].apply(analisar_sentimento)
    df['sentimento_ambiente'] = df['enviroment_comment'].apply(analisar_sentimento)

    # Cálculo da categoria NPS com base na nota de satisfação
    df['categoria_nps'] = df['satisfaction_rating'].apply(calcula_nps)

    return df

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