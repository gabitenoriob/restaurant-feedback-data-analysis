# Função para analisar o sentimento do comentário
from textblob import TextBlob


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