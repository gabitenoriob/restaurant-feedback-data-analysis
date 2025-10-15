import pandas as pd
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

def calcular_nps(df):
    # Features numéricas (exceto a recomendação)
    df_processado = df.copy()
    sentiment_map = {'Positivo': 1, 'Neutro': 0, 'Negativo': -1, 'Misto': 0}
    
    df_processado['food_rating'] = df_processado['sentimento_comida'].map(sentiment_map).fillna(0)
    df_processado['service_rating'] = df_processado['sentimento_servico'].map(sentiment_map).fillna(0)
    df_processado['environment_rating'] = df_processado['sentimento_ambiente'].map(sentiment_map).fillna(0)
    features = ['service_rating', 'food_rating', 'environment_rating']  
    X = df_processado[features]
    y = df_processado['recommendation_rating']

    # Treinar o modelo de regressão linear
    model = LinearRegression()
    model.fit(X, y)

    # Exibir os coeficientes
    print("Impacto de cada aspecto na nota de recomendação:")
    for feature, coef in zip(features, model.coef_):
        print(f"Cada ponto a mais em {feature} aumenta {coef:.2f} pontos na recomendação.")

    plt.bar(features, model.coef_)
    plt.ylabel('Impacto na recomendação')
    plt.title('Coeficientes da Regressão Linear')
    plt.show()

    df_processado['nps_pred'] = model.predict(X)
    df_processado['recommendation_rating'] = df_processado['recommendation_rating'].fillna(df_processado['nps_pred'])
    df_processado.to_csv("nps_predictions.csv", index=False)
    return df_processado['nps_pred']
