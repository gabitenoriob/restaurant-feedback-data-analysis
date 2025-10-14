import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump # Opcional: para salvar o modelo treinado

def medicao_churn(df):
   
    df_processado = df.copy()

    # Define a variável alvo (target) com base na nota.
    # Clientes detratores (nota <= 6) são considerados churn = 1.
    df_processado['churn'] = np.where(df_processado['recommendation_rating'] <= 6, 1, 0)

    sentiment_map = {'Positivo': 1, 'Neutro': 0, 'Negativo': -1, 'Misto': 0}
    
    df_processado['food_rating'] = df_processado['sentimento_comida'].map(sentiment_map).fillna(0)
    df_processado['service_rating'] = df_processado['sentimento_servico'].map(sentiment_map).fillna(0)
    df_processado['environment_rating'] = df_processado['sentimento_ambiente'].map(sentiment_map).fillna(0)

    # Define a lista de features (variáveis) que o modelo usará para aprender
    features = [
        'service_rating', 'food_rating', 'environment_rating',
        'sentimento_comida', 'sentimento_ambiente', 'sentimento_servico',
        'categoria_nps',
        'nome_topico', 'probabilidade_topico'
    ]
    target = 'churn'

    # Prepara os dados de entrada (X) e saída (y)
    # pd.get_dummies converte colunas de texto (categóricas) em colunas numéricas
    X = pd.get_dummies(df_processado[features], drop_first=True)
    y = df_processado[target]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    # Avalia a performance do modelo no conjunto de teste
    y_pred = rf.predict(X_test)
    print("--- Avaliação do Modelo ---")
    print(f"Acurácia no conjunto de teste: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    # Opcional: Salva o modelo treinado para uso futuro
    dump(rf, 'modelo_churn.joblib')
    print("\nModelo salvo como 'modelo_churn.joblib'")
    
    # --- Previsão para Todos os Dados ---
   
    X_all = pd.get_dummies(df_processado[features], drop_first=True)
    X_all = X_all.reindex(columns=X_train.columns, fill_value=0)
    
    # Adiciona as colunas de previsão ao DataFrame
    df_processado['churn_pred'] = rf.predict(X_all)
    df_processado['churn_prob'] = rf.predict_proba(X_all)[:, 1] # Probabilidade de ser churn=1

    print("\nPrevisão de churn concluída para todos os dados.")
    return df_processado