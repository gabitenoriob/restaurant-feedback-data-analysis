import pandas as pd
import nltk
import spacy
from transformers import pipeline
import json
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from limpeza_dados import limpar_dados

nltk.download('punkt')
nltk.download('punkt_tab', quiet=True)

print("Carregando modelo BERT...")
#baixa o modelo do huggingface 
sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    #model="distilbert-base-multilingual-cased"
    model = "nlptown/bert-base-multilingual-uncased-sentiment"
)

print("Carregando modelo spaCy...")
nlp_spacy = spacy.load("pt_core_news_lg")

print("Configuração concluída!")

ASPECT_KEYWORDS = { "comida": [ "comida", "prato", "pratos", "refeição", "sabor", "sabores", "cardápio", "menu", "picanha", "moqueca", "sobremesa", "entrada", "porção", "lanche", "almoço", "jantar", "café", "bebida", "drinque", "coquetel", "aperitivo", "massa", "carne", "peixe", "frango", "salada", "sopa", "tempero", "temperado", "salgado", "doce", "gostoso", "delicioso", "saboroso", "fresco", "quentinho", "frio", "bem servido", "mal passado", "ponto da carne", "grelhado", "assado", "frito", "cru", "vegano", "vegetariano", "porção generosa", "apresentação", "montagem", "textura", "cheiro", "aroma" ], "servico": [ "serviço", "atendimento", "atendente", "garçom", "garçonete", "equipe", "demora", "rapidez", "atencioso", "grosseiro", "educado", "prestativo", "simpático", "gentil", "cortês", "mal educado", "eficiente", "ineficiente", "lento", "demorado", "organizado", "desorganizado", "profissional", "amador", "receptivo", "indelicado", "disponível", "solícito", "comunicação", "pedido", "erro no pedido", "acertaram o pedido", "demoraram muito", "pronto atendimento", "resposta rápida", "falta de atenção", "cordial", "respeitoso", "atraso", "tratamento", "garçom sumiu", "garçom rápido", "educação", "boa vontade", "resolução", "problema resolvido", "cuidado com cliente" ], "ambiente": [ "ambiente", "lugar", "espaço", "decoração", "música", "barulho", "limpeza", "confortável", "aconchegante", "barulhento", "tranquilo", "agradável", "bonito", "iluminação", "claridade", "escuridão", "lotado", "vazio", "organização", "bagunçado", "ventilação", "ar condicionado", "cheiro", "odor", "vista", "paisagem", "jardim", "terraço", "varanda", "interno", "externo", "mesas", "cadeiras", "banheiro", "banheiros limpos", "higiene", "toalete", "ambiente familiar", "romântico", "sofisticado", "moderno", "rústico", "agradável para conversar", "decorado", "climatizado", "aconchego", "energia do lugar", "atmosfera", "vibe", "música alta", "música ambiente" ], "geral": ["recomendo", "não recomendo", "voltarei", "não voltarei", "experiencia ótima", "experiencia ruim", "experiencia maravilhosa", "experiencia péssima", "vale a pena", "não vale a pena", "custo benefício", "preço justo", "caro", "barato", "muito caro", "muito barato", "promoção", "desconto", "oferta", "custo benefício"] }



def extrair_frases_por_aspecto(texto, aspecto_keywords):
    frases_relevantes = []
    sentencas = sent_tokenize(texto, language="portuguese") #divide em frases
    for sentenca in sentencas:
        if any(keyword in sentenca.lower() for keyword in aspecto_keywords): #avalia se a frase tem alguma keyword
            frases_relevantes.append(sentenca)
    print(f"frases relevantes para o aspecto '{aspecto_keywords}': {frases_relevantes}")
    return frases_relevantes

def analisar_sentimento_bert(frase):
    resultado = sentiment_pipeline(frase[:512])[0]
    label = resultado['label']
    label_map = {"LABEL_0": "Negativo", "LABEL_1": "Neutro", "LABEL_2": "Positivo"}
    print(f"Análise de sentimento para a frase '{frase}': {label_map.get(label, 'Desconhecido')}")
    return label_map.get(label, "Desconhecido")

def extrair_justificativa_spacy(frase, aspecto_keywords): #extrai adjetivos que justificam as notas
    doc = nlp_spacy(frase)
    justificativas = set()
    for token in doc:
        if token.lemma_.lower() in aspecto_keywords:
            for child in token.children: #exemplo comida deliciosa comida = substantivo e deliciosa = adjetivo
                if child.pos_ == 'ADJ':
                    justificativas.add(child.text)
            if token.head.pos_ == 'ADJ':
                justificativas.add(token.head.text)
    if not justificativas:
        justificativas.update([token.text for token in doc if token.pos_ == 'ADJ'])
    print(f"Justificativas encontradas para a frase '{frase}': {justificativas}")
    return list(justificativas)

def analisar_comentario_completo(comentario): #Orquestra tudo para um único comentário.
    if not isinstance(comentario, str) or not comentario.strip():
        return {}
    
    comentario_limpo = limpar_texto(comentario)
    resultados_finais = {}
    
    for aspecto, keywords in ASPECT_KEYWORDS.items():
        frases_relevantes = extrair_frases_por_aspecto(comentario_limpo, keywords)
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
        else:
            return "Detrator"
    except (ValueError, TypeError):
        return "Desconhecido"

def agrupar_comentarios_por_tema(df, coluna_comentario="general_comment"):
    modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")
    textos = df[coluna_comentario].astype(str).apply(limpar_texto).tolist()
    embeddings = modelo_embeddings.encode(textos, show_progress_bar=True)
    topic_model = BERTopic(language="portuguese", verbose=True)
    topics, probs = topic_model.fit_transform(textos, embeddings)
    df["topico"] = topics
    return df, topic_model

def transformar_dados(df, coluna_comentario='general_comment', coluna_nota='recommendation_rating'):
    print("Iniciando a transformação dos dados...")
    if coluna_comentario not in df.columns or coluna_nota not in df.columns:
        raise ValueError(f"O DataFrame precisa ter as colunas '{coluna_comentario}' e '{coluna_nota}'")
    
    # Análise de sentimentos por aspecto
    resultados_analise = df[coluna_comentario].apply(analisar_comentario_completo)
    df_analise = pd.json_normalize(resultados_analise)

    # Extrair colunas de sentimento
    sentimento_cols = [col for col in df_analise.columns if col.endswith(".sentimento")]
    df_analise = df_analise[sentimento_cols]
    df_analise.columns = [f"sentimento_{col.split('.')[0]}" for col in sentimento_cols]

    # Garantir que todos os aspectos estejam presentes
    for aspecto in ASPECT_KEYWORDS.keys():
        col_name = f"sentimento_{aspecto}"
        if col_name not in df_analise.columns:
            df_analise[col_name] = "Não"

    df_final = pd.concat([df, df_analise], axis=1)

    # Adiciona NPS
    df_final['categoria_nps'] = df_final[coluna_nota].apply(calcula_nps)

    # Agrupamento de temas
    print("Agrupando comentários por tema...")
    df_final, topic_model = agrupar_comentarios_por_tema(df_final, coluna_comentario)
    print("Transformação concluída!")
    return df_final, topic_model

# =========================================
# 🧪 TESTE COM DADOS DE EXEMPLO
# =========================================
if __name__ == "__main__":
    data = {
        "general_comment": [
            "A comida estava ótima, mas o ambiente era sujo.",
            "O atendimento foi excelente e o preço justo.",
            "Demorou muito para chegar, comida fria e ruim.",
            "Ambiente agradável e comida saborosa, recomendo!",
            "Preço alto e porção pequena, não vale a pena."
        ],
        "recommendation_rating": [3, 5, 1, 5, 2]
    }
    df_exemplo = pd.DataFrame(data)
    resultado, modelo_topicos = transformar_dados(df_exemplo)
    print(resultado)
    print("\n📊 Principais tópicos encontrados:")
    print(modelo_topicos.get_topic_info().head())
