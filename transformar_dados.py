import pandas as pd
import spacy
from transformers import pipeline
import re
from collections import defaultdict

print("Carregando modelo de sentimento (BERT)...")
# O modelo nlptown é ótimo para essa tarefa de classificação em 5 estrelas.
SENTIMENT_PIPELINE = pipeline(
    task="sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

print("Carregando modelo de linguagem (spaCy)...")
NLP_SPACY = spacy.load("pt_core_news_lg")


ASPECT_KEYWORDS = {
    "comida": [
        "comida", "prato", "sabor", "gosto", "cardápio", "menu", "bebida", "drink", "sobremesa", "almoço",
        "janta", "jantar", "lanche", "carne", "peixe", "massa", "feijoada", "moqueca", "picanha", "sopa",
        "salada", "porção", "entrada", "acompanhamento", "refeição", "temperatura", "temperado", "tempero"
    ],
    "servico": [
        "serviço", "atendimento", "garçom", "garçonete", "equipe", "pedido", "entrega", "demora", "espera",
        "rapidez", "atendente", "funcionário", "funcionária", "staff", "recepcionista", "tratamento", "prestador"
    ],
    "ambiente": [
        "ambiente", "local", "lugar", "espaço", "decoração", "música", "barulho", "som", "iluminação", "limpeza",
        "banheiro", "conforto", "cadeira", "mesa", "ar condicionado", "ventilação", "vista", "paisagem", "clima"
    ],
    "preco": [
        "preço", "valor", "custo", "conta", "cobrança", "custo-benefício", "caro", "barato", "promoção",
        "taxa", "gorjeta", "desconto"
    ]
}


SENTIMENT_MODIFIERS = {
    "comida": {
        "Positivo": [
            "delicioso", "deliciosa", "deliciosos", "deliciosas", "saboroso", "saborosa", "excelente", "ótimo", "ótima",
            "maravilhoso", "maravilhosa", "perfeito", "perfeita", "incrível", "gostoso", "gostosa", "fresco", "fresca",
            "quentinho", "quentinha", "bem servido", "bem servida", "porção generosa", "bem temperado", "temperado no ponto",
            "divino", "divina", "dos deuses", "top", "show de bola", "bem servido", "porção grande"
        ],
        "Negativo": [
            "ruim", "péssimo", "péssima", "horrível", "frio", "fria", "gelado", "sem sabor", "sem gosto", "insosso",
            "sem graça", "queimado", "queimada", "cru", "crua", "seco", "seca", "gorduroso", "gordurosa", "oleoso",
            "oleosa", "estragado", "velho", "velha", "mal cozido", "mal passada", "salgado demais", "doce demais",
            "azedo", "amargo", "requentado", "porção pequena", "comida sem sabor"
        ]
    },
    "servico": {
        "Positivo": [
            "rápido", "rápida", "eficiente", "atencioso", "atenciosa", "educado", "educada", "simpático", "simpática",
            "prestativo", "prestativa", "solícito", "solícita", "cordial", "gentil", "ágil", "bom atendimento",
            "super atencioso", "muito educado", "ótimo serviço"
        ],
        "Negativo": [
            "lento", "lenta", "demorou", "demorado", "demorada", "grosseiro", "grossa", "mal educado", "mal educada",
            "desatento", "desatenta", "confuso", "errado", "esqueceram", "sumiu", "atendimento ruim", "pouco atencioso",
            "demora absurda", "mal treinado", "negligente", "não gostei do atendimento"
        ]
    },
    "ambiente": {
        "Positivo": [
            "agradável", "aconchegante", "confortável", "limpo", "bonito", "tranquilo", "arejado", "moderno",
            "organizado", "bem decorado", "agradável de ficar", "clima bom", "vista bonita", "espaçoso"
        ],
        "Negativo": [
            "sujo", "barulhento", "apertado", "desconfortável", "quente", "abafado", "mal cheiro", "malcheiroso",
            "mal iluminado", "escuro", "velho", "feio", "desorganizado", "caótico"
        ]
    },
    "preco": {
        "Positivo": [
            "barato", "barata", "acessível", "justo", "justa", "bom preço", "preço bom", "vale a pena", "ótimo custo benefício"
        ],
        "Negativo": [
            "caro", "cara", "caríssimo", "caríssima", "absurdo", "exagerado", "não vale", "roubo", "exploração",
            "preço alto", "cobrança indevida","preço alto"
        ]
    },
    "geral": {
        "Positivo": [
            "recomendo", "recomendo muito", "voltarei", "voltaria", "perfeito", "maravilhoso", "incrível", "vale a pena",
            "excelente", "amei", "top", "show", "ótimo lugar", "da hora", "sensacional", "nota dez", "tudo ótimo"
        ],
        "Negativo": [
            "decepcionante", "não recomendo", "não volto", "evitem", "terrível", "horrível", "péssimo", "nunca mais",
            "lixo", "fracasso", "horrendo", "experiência ruim", "não vale a pena"
        ]
    }
}


NEGATIONS = [
    "não", "nao", "nunca", "jamais", "sem", "nem", "tampouco", "de forma alguma", "de jeito nenhum"
]

ALL_SENTIMENT_MODIFIERS = {
    "Positivo": list(set([kw for aspect in SENTIMENT_MODIFIERS.values() for kw in aspect.get("Positivo", [])])),
    "Negativo": list(set([kw for aspect in SENTIMENT_MODIFIERS.values() for kw in aspect.get("Negativo", [])]))
}


class AspectBasedSentimentAnalyzer:
    def __init__(self, nlp_model, aspect_keywords, sentiment_modifiers, negations):
        self.nlp = nlp_model
        self.sentiment_modifiers = sentiment_modifiers
        self.negations = negations
        self._keyword_to_aspect_map = {
            self.nlp(kw)[0].lemma_: aspect 
            for aspect, kws in aspect_keywords.items() 
            for kw in kws
        }

    def _get_sentiment_from_phrase(self, phrase):
        """Analisa uma pequena frase (chunk) e retorna o sentimento e a palavra que o definiu."""
        text_lower = phrase.lower()
        is_negated = any(f"{neg} " in text_lower for neg in self.negations)

        # Procura por modificadores negativos primeiro, pois costumam ter mais peso
        for keyword in self.sentiment_modifiers["Negativo"]:
            if keyword in text_lower:
                return "Positivo" if is_negated else "Negativo", keyword
        
        for keyword in self.sentiment_modifiers["Positivo"]:
            if keyword in text_lower:
                return "Negativo" if is_negated else "Positivo", keyword
        
        return "Neutro", None

    def analyze(self, comment):
        if not isinstance(comment, str) or not comment.strip():
            return {}
        
        doc = self.nlp(comment)
        aspect_sentiments = defaultdict(list)

        for sent in doc.sents:
            found_aspects_in_sent = {}
            # Primeiro, encontra todos os aspectos na sentença
            for token in sent:
                aspect = self._keyword_to_aspect_map.get(token.lemma_)
                if aspect:
                    found_aspects_in_sent[token.i] = (aspect, token) # Salva o índice do token do aspecto

            if not found_aspects_in_sent:
                continue

            # Agora, para cada palavra na sentença, verifica se é uma opinião
            for token in sent:
                sentiment, modifier = self._get_sentiment_from_phrase(token.text)
                
                if sentiment != "Neutro":
                    # Se encontrou uma opinião, procura o aspecto mais próximo a ela
                    closest_aspect_token = None
                    min_distance = float('inf')
                    
                    for aspect_idx, (aspect, aspect_token) in found_aspects_in_sent.items():
                        distance = abs(token.i - aspect_idx)
                        if distance < min_distance:
                            min_distance = distance
                            closest_aspect_token = aspect_token
                    
                    if closest_aspect_token:
                        aspect_name = self._keyword_to_aspect_map.get(closest_aspect_token.lemma_)
                        # Constrói uma justificativa mais completa (ex: "ambiente sujo")
                        justification = f"{closest_aspect_token.text} {modifier}" if min_distance <= 2 else modifier
                        aspect_sentiments[aspect_name].append({
                            "sentimento": sentiment,
                            "justificativa": justification
                        })

        final_results = {}
        for aspect, opinions in aspect_sentiments.items():
            if not opinions: continue
            
            sentimentos = [op['sentimento'] for op in opinions]
            justificativas = [op['justificativa'] for op in opinions]
            
            if "Positivo" in sentimentos and "Negativo" in sentimentos: final_sentiment = "Misto"
            elif "Negativo" in sentimentos: final_sentiment = "Negativo"
            else: final_sentiment = "Positivo"

            final_results[aspect] = {
                "sentimento": final_sentiment,
                "justificativa": sorted(list(set(justificativas)), key=len, reverse=True)
            }
        return final_results

def transformar_dados(df, coluna_comentario='general_comment'):
    print("Iniciando a transformação dos dados com a lógica CORRIGIDA...")
    analyzer = AspectBasedSentimentAnalyzer(
        nlp_model=NLP_SPACY,
        aspect_keywords=ASPECT_KEYWORDS,
        sentiment_modifiers=ALL_SENTIMENT_MODIFIERS, 
        negations=NEGATIONS
    )
    resultados_analise = df[coluna_comentario].apply(analyzer.analyze)
    df_analise = pd.json_normalize(resultados_analise)
    df_final = df.copy()
    all_aspects = list(ASPECT_KEYWORDS.keys())
    
    aspect_sentiment_cols = [] 
    
    for aspecto in all_aspects:
        col_sentimento = f"sentimento_{aspecto}"
        col_justificativa = f"justificativa_{aspecto}"
        aspect_sentiment_cols.append(col_sentimento)

        if f"{aspecto}.sentimento" in df_analise.columns:
            df_final[col_sentimento] = df_analise[f"{aspecto}.sentimento"]
        else:
            df_final[col_sentimento] = "Não Mencionado"
        
        if f"{aspecto}.justificativa" in df_analise.columns:
            df_final[col_justificativa] = df_analise[f"{aspecto}.justificativa"].fillna("").apply(list)
        else:
            df_final[col_justificativa] = [[] for _ in range(len(df_final))]

    def calcular_sentimento_geral(row):
        sentimentos = [row[col] for col in aspect_sentiment_cols if row[col] != "Não Mencionado"]
        
        if not sentimentos:
            return "Indefinido"
        
        sentimentos_set = set(sentimentos)
        
        if "Positivo" in sentimentos_set and "Negativo" in sentimentos_set:
            return "Misto"
        if "Negativo" in sentimentos_set:
            return "Negativo"
        if "Positivo" in sentimentos_set:
            return "Positivo"
        if "Neutro" in sentimentos_set:
            return "Neutro"
        
        return "Indefinido"

    df_final['sentimento_geral'] = df_final.apply(calcular_sentimento_geral, axis=1)

    print("Transformação concluída!")
    return df_final

if __name__ == "__main__":
    data = {
        "general_comment": [
            "A comida estava ótima, mas o ambiente era sujo.",
            "O atendimento foi excelente e o preço justo.",
            "Demorou muito para chegar, comida fria e ruim.",
            "Ambiente agradável e comida saborosa, recomendo!",
            "Preço alto e porção pequena, não vale a pena.",
            "O garçom foi muito educado, mas a música estava alta demais.",
            "A sobremesa estava deliciosa, voltarei com certeza.",
            "O banheiro estava limpo e o ambiente aconchegante.",
            "A carne veio mal passada, pedi bem passada.",
            "O restaurante estava lotado e o serviço foi lento.",
            "Comida sem sabor algum.",
            "Não gostei do atendimento.",
        ],
        "recommendation_rating": [4, 10, 1, 9, 2, 6, 10, 8, 3, 3, 2, 2]
    }
    df_exemplo = pd.DataFrame(data)
    
    resultado = transformar_dados(df_exemplo)

    
    colunas_para_exibir = [
        "general_comment", 
        "sentimento_comida", "justificativa_comida",
        "sentimento_servico", "justificativa_servico",
        "sentimento_ambiente", "justificativa_ambiente",
        "sentimento_preco", "justificativa_preco", 'sentimento_geral'
    ]
    
    for col in colunas_para_exibir:
        if col not in resultado.columns:
            resultado[col] = "Não Mencionado" if "sentimento" in col else [[] for _ in range(len(resultado))]

    print("\n--- RESULTADOS DA ANÁLISE CORRIGIDA ---")
    print(resultado[colunas_para_exibir].to_string())
    
    resultado.to_csv("resultados_absa_corrigido.csv", index=False, encoding='utf-8-sig')
    print("\nResultados salvos em 'resultados_absa_corrigido.csv'")