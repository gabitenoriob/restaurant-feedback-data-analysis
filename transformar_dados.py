import pandas as pd
import nltk
import spacy
from transformers import pipeline
import json # Importado para visualização

# --- 1. CONFIGURAÇÃO INICIAL E CARREGAMENTO DOS MODELOS ---

print("Carregando modelo BERT...")
# Carrega o pipeline de sentimento do Hugging Face
sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

print("Carregando modelo spaCy...")
# Carrega o modelo de linguagem em português do spaCy
nlp_spacy = spacy.load("pt_core_news_lg")

print("Carregando NLTK...")
# Baixa o tokenizador de sentenças do NLTK (se necessário)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

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
    "serviço": [
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
    ]
}

# --- 3. DEFINIÇÃO DAS FUNÇÕES AUXILIARES DE ANÁLISE ---

def extrair_frases_por_aspecto(texto, aspecto_keywords):
    frases_relevantes = []
    sentencas = nltk.sent_tokenize(texto, language='portuguese')
    for sentenca in sentencas:
        if any(keyword in sentenca.lower() for keyword in aspecto_keywords):
            frases_relevantes.append(sentenca)
    return frases_relevantes

def analisar_sentimento_bert(frase):
    resultado = sentiment_pipeline(frase)[0]
    estrelas = int(resultado['label'].split(' ')[0])
    
    if estrelas >= 4:
        return "Positivo"
    elif estrelas == 3:
        return "Neutro"
    else:
        return "Negativo"

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
    """Orquestra a análise de um único comentário, retornando um dicionário com os resultados."""
    # Garante que o comentário é uma string e não está vazio
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


def transformar_dados(df, coluna_comentario='general_comment', coluna_nota='satisfaction_rating'):
    """
    Função principal que aplica a análise de sentimentos e o cálculo de NPS a um DataFrame.
    """
    print("Iniciando a transformação dos dados...")

    # Garante que as colunas de entrada existem
    if coluna_comentario not in df.columns or coluna_nota not in df.columns:
        raise ValueError(f"O DataFrame precisa ter as colunas '{coluna_comentario}' e '{coluna_nota}'")
    
    # 1. Aplica a análise de sentimentos baseada em aspectos
    # O .apply executa a função 'analisar_comentario_completo' para cada linha
    resultados_analise = df[coluna_comentario].apply(analisar_comentario_completo)

    # 2. Converte a série de dicionários em um DataFrame
    df_analise = pd.json_normalize(resultados_analise)
    print("Resultados da análise (exemplo):")
    print(df_analise.head().to_json(orient='records', force_ascii=False, indent=2))  

    # 3. Renomeia as colunas para um formato mais limpo (ex: 'comida.sentimento' -> 'sentimento_comida')
    df_analise.columns = [f"{col.split('.')[1]}_{col.split('.')[0]}" for col in df_analise.columns]

    # 4. Junta o DataFrame original com os resultados da análise
    df_final = pd.concat([df, df_analise], axis=1)

    # 5. Calcula e adiciona a coluna de categoria NPS
    df_final['categoria_nps'] = df_final[coluna_nota].apply(calcula_nps)
    
    print("Transformação concluída.")
    return df_final

