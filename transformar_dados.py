import pandas as pd
import nltk
import spacy
from transformers import pipeline
import json # Importado para visualização
import nltk

nltk.download('punkt')      # necessário
nltk.download('punkt_tab')  # tokenização tabular em português

from nltk.tokenize import word_tokenize


# --- 1. CONFIGURAÇÃO INICIAL E CARREGAMENTO DOS MODELOS ---

print("Carregando modelo BERT...")
# Carrega o pipeline de sentimento do Hugging Face
sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-multilingual-cased"
    #"nlptown/bert-base-multilingual-uncased-sentiment"
)

print("Carregando modelo spaCy...")
# Carrega o modelo de linguagem em português do spaCy
nlp_spacy = spacy.load("pt_core_news_lg")

print("Carregando NLTK...")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Baixando recurso NLTK 'punkt'...")
    nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize

def tokenize_portuguese(texto):
    return word_tokenize(texto, language="portuguese")

def sent_tokenize_portuguese(texto):
    return sent_tokenize(texto, language="portuguese")


print("Configuração concluída!")

# --- 2. DEFINIÇÃO DOS ASPECTOS E PALAVRAS-CHAVE ---

ASPECT_KEYWORDS = {
    "comida": [
        "comida", "prato", "pratos", "refeição", "sabor", "sabores", "cardápio", "menu", "picanha", "moqueca", "sobremesa",
        "entrada", "porção", "lanche", "almoço", "jantar", "café", "bebida", "drinque", "coquetel", "aperitivo", "massa",
        "carne", "peixe", "frango", "salada", "sopa", "tempero", "temperado", "salgado", "doce", "gostoso", "delicioso",
        "saboroso", "fresco", "quentinho", "frio", "bem servido", "mal passado", "ponto da carne", "grelhado", "assado",
        "frito", "cru", "vegano", "vegetariano", "porção generosa", "apresentação", "montagem", "textura", "cheiro", "aroma"
    ],
    "servico": [
        "serviço", "atendimento", "atendente", "garçom", "garçonete", "equipe", "demora", "rapidez", "atencioso",
        "grosseiro", "educado", "prestativo", "simpático", "gentil", "cortês", "mal educado", "eficiente", "ineficiente",
        "lento", "demorado", "organizado", "desorganizado", "profissional", "amador", "receptivo", "indelicado",
        "disponível", "solícito", "comunicação", "pedido", "erro no pedido", "acertaram o pedido", "demoraram muito",
        "pronto atendimento", "resposta rápida", "falta de atenção", "cordial", "respeitoso", "atraso", "tratamento",
        "garçom sumiu", "garçom rápido", "educação", "boa vontade", "resolução", "problema resolvido", "cuidado com cliente"
    ],
    "ambiente": [
        "ambiente", "lugar", "espaço", "decoração", "música", "barulho", "limpeza", "confortável", "aconchegante",
        "barulhento", "tranquilo", "agradável", "bonito", "iluminação", "claridade", "escuridão", "lotado", "vazio",
        "organização", "bagunçado", "ventilação", "ar condicionado", "cheiro", "odor", "vista", "paisagem", "jardim",
        "terraço", "varanda", "interno", "externo", "mesas", "cadeiras", "banheiro", "banheiros limpos", "higiene",
        "toalete", "ambiente familiar", "romântico", "sofisticado", "moderno", "rústico", "agradável para conversar",
        "decorado", "climatizado", "aconchego", "energia do lugar", "atmosfera", "vibe", "música alta", "música ambiente"
    ],
    "geral": ["recomendo", "não recomendo", "voltarei", "não voltarei", "experiencia ótima", "experiencia ruim", "experiencia maravilhosa", "experiencia péssima", "vale a pena", "não vale a pena", "custo benefício", "preço justo", "caro", "barato", "muito caro", "muito barato", "promoção", "desconto", "oferta", "custo benefício"] 
}

# --- 3. DEFINIÇÃO DAS FUNÇÕES AUXILIARES DE ANÁLISE ---

def extrair_frases_por_aspecto(texto, aspecto_keywords):
    frases_relevantes = []
    sentencas = sent_tokenize_portuguese(texto)
    for sentenca in sentencas:
        if any(keyword in sentenca.lower() for keyword in aspecto_keywords):
            frases_relevantes.append(sentenca)
    return frases_relevantes

# def analisar_sentimento_bert(frase):
#     resultado = sentiment_pipeline(frase)[0]
#     estrelas = int(resultado['label'].split(' ')[0])
    
#     if estrelas >= 4:
#         return "Positivo"
#     elif estrelas == 3:
#         return "Neutro"
#     else:
#         return "Negativo"

def analisar_sentimento_bert(frase):
    resultado = sentiment_pipeline(frase)[0]
    label = resultado['label']

    # Mapeamento dos labels do modelo para sentimento
    label_map = {
        "LABEL_0": "Negativo",
        "LABEL_1": "Neutro",
        "LABEL_2": "Positivo"
    }

    return label_map.get(label, "Desconhecido")


def extrair_justificativa_spacy(frase, aspecto_keywords):
    doc = nlp_spacy(frase)
    justificativas = set()
    for token in doc:
        if token.lemma_.lower() in aspecto_keywords:
            for child in token.children:
                if child.pos_ == 'ADJ':
                    justificativas.add(child.text)
            if token.head.pos_ == 'ADJ':
                justificativas.add(token.head.text)
    if not justificativas:
        justificativas.update([token.text for token in doc if token.pos_ == 'ADJ'])
    return list(justificativas)

def analisar_comentario_completo(comentario):
    if not isinstance(comentario, str) or not comentario.strip():
        return {}
    
    resultados_finais = {}
    
    for aspecto, keywords in ASPECT_KEYWORDS.items():
        frases_relevantes = extrair_frases_por_aspecto(comentario, keywords)
        if not frases_relevantes:
            continue
        
        sentimentos_aspecto = [analisar_sentimento_bert(frase) for frase in frases_relevantes]
        justificativas_aspecto = []
        
        for frase in frases_relevantes:
            justificativas_aspecto.extend(extrair_justificativa_spacy(frase, keywords))
        
        sentimento_geral = "Neutro"
        if "Positivo" in sentimentos_aspecto and "Negativo" in sentimentos_aspecto:
            sentimento_geral = "Misto"
        elif sentimentos_aspecto:
            sentimento_geral = max(set(sentimentos_aspecto), key=sentimentos_aspecto.count)
        
        resultados_finais[aspecto] = {
            "sentimento": sentimento_geral,
            "justificativa": list(set(justificativas_aspecto))
        }
    
    return resultados_finais

def calcula_nps(nota):
    try:
        nota = int(nota)
        if nota >= 9:
            return "Promotor"
        elif 7 <= nota <= 8:
            return "Passivo"
        else: # Cobrindo de 0 a 6
            return "Detrator"
    except (ValueError, TypeError):
        return "Desconhecido"


def transformar_dados(df, coluna_comentario='general_comment', coluna_nota='recommendation_rating'):
    print("Iniciando a transformação dos dados...")
    print(f"Colunas do DataFrame: {df.columns.tolist()}")
    if coluna_comentario not in df.columns or coluna_nota not in df.columns:
        raise ValueError(f"O DataFrame precisa ter as colunas '{coluna_comentario}' e '{coluna_nota}'")
    
    resultados_analise = df[coluna_comentario].apply(analisar_comentario_completo)
    df_analise = pd.json_normalize(resultados_analise)

    sentimento_cols = [col for col in df_analise.columns if col.endswith(".sentimento")]
    df_analise = df_analise[sentimento_cols]

    df_analise.columns = [f"sentimento_{col.split('.')[0]}" for col in sentimento_cols]

    # Garantir colunas vazias preenchidas com "Neutro"
    for aspecto in ASPECT_KEYWORDS.keys():
        col_name = f"sentimento_{aspecto}"
        if col_name not in df_analise.columns:
            df_analise[col_name] = "Não"

    df_final = pd.concat([df, df_analise], axis=1)

    # Adiciona coluna geral baseada no aspecto "geral"
    if "sentimento_geral" in df_analise.columns:
        df_final["sentimento_geral"] = df_analise["sentimento_geral"]
    else:
        df_final["sentimento_geral"] = "Não"

    df_final['categoria_nps'] = df_final[coluna_nota].apply(calcula_nps)
    print(df_final.columns.tolist())

    print("Transformação concluída.")
    return df_final

